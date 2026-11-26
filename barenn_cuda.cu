// Versão GPU (CUDA) do treino do Barenn.
//
// O algoritmo é o mesmo SGD por amostra da versão CPU, com os mesmos dados, a
// mesma inicialização e a mesma ordem de operações de ponto flutuante em cada
// neurônio.
//
// Kernel persistente: o treino inteiro (todas as épocas e amostras) roda num
// único lançamento, com um único bloco de threads. As etapas de cada amostra
// (forward das 3 camadas, perda, delta, gradiente da entrada e atualização)
// são separadas por __syncthreads(), que custa dezenas de ciclos, em vez de
// um lançamento de kernel por etapa (~µs). Um só bloco é necessário porque o
// SGD por amostra exige sincronização entre etapas, e __syncthreads() só
// sincroniza threads do mesmo bloco.
//
// A leitura dos dados, a inicialização das camadas e a avaliação reaproveitam
// o código da CPU (barenn.hpp). Os pesos treinados são copiados de volta para
// as camadas da CPU, que fazem a avaliação.
//
// Compilar com --fmad=false: sem isso o nvcc funde a*b+c em FMA, o que muda o
// arredondamento e a saída deixa de ser idêntica à referência.

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "barenn.hpp"

using namespace std;

// Aborta com mensagem clara se uma chamada da API CUDA falhar.
#define CUDA_CHECK(chamada)                                                   \
    do {                                                                      \
        cudaError_t erro_cuda = (chamada);                                    \
        if (erro_cuda != cudaSuccess) {                                       \
            fprintf(stderr, "Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(erro_cuda));                           \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

// Threads do bloco único do kernel persistente. Pode ser trocado na
// compilação (-DTHREADS=...) para a varredura de tamanho de bloco.
#ifndef THREADS
#define THREADS 1024
#endif

namespace {

// Parâmetros e buffers de uma camada na memória global da GPU. Layout dos
// pesos igual ao da CPU: w[j * saidas + i] liga a entrada j ao neurônio i.
struct CamadaDev {
    float* w;      // pesos (entradas x saídas, transpostos)
    float* b;      // vieses
    float* z;      // soma antes da ativação (último forward)
    float* a;      // saída após a ativação
    float* delta;  // gradiente antes da ativação
    float* grad;   // gradiente da entrada
    int entradas;
    int saidas;
    bool final;
};

// ---------------------------------------------------------------------------
// Etapas de uma camada, executadas pelas THREADS threads do bloco. Cada
// função distribui o trabalho em passos de THREADS (grid-stride dentro do
// bloco). Quem chama é responsável pelo __syncthreads() entre etapas.
// ---------------------------------------------------------------------------

// z = b + x·W e a = ReLU(z) (ou a = z na última camada). Uma thread por
// neurônio; a soma de cada neurônio segue a mesma ordem da CPU (viés
// primeiro, depois j = 0, 1, ...).
__device__ void forward(const CamadaDev& c, const float* x) {
    for (int i = threadIdx.x; i < c.saidas; i += THREADS) {
        float soma = c.b[i];
        for (int j = 0; j < c.entradas; ++j) {
            soma += x[j] * c.w[j * c.saidas + i];
        }
        c.z[i] = soma;
        c.a[i] = c.final ? soma : fmaxf(0.0f, soma);
    }
}

// delta = gradiente da perda antes da ativação (derivada da ReLU).
__device__ void calcula_delta(const CamadaDev& c, const float* grad_saida) {
    for (int i = threadIdx.x; i < c.saidas; i += THREADS) {
        c.delta[i] = c.final ? grad_saida[i]
                             : grad_saida[i] * (c.z[i] > 0.0f ? 1.0f : 0.0f);
    }
}

// Gradiente da entrada, com os pesos anteriores à atualização. Uma thread
// por entrada, somando os neurônios na mesma ordem da CPU.
__device__ void grad_entrada(const CamadaDev& c) {
    for (int j = threadIdx.x; j < c.entradas; j += THREADS) {
        float soma = 0;
        for (int i = 0; i < c.saidas; ++i) {
            soma += c.delta[i] * c.w[j * c.saidas + i];
        }
        c.grad[j] = soma;
    }
}

// SGD: w -= taxa * delta_i * x_j e b -= taxa * delta_i. Índices consecutivos
// de thread acessam pesos consecutivos (acesso coalescido).
__device__ void atualiza(const CamadaDev& c, const float* x, float taxa) {
    const int n_pesos = c.entradas * c.saidas;
    for (int k = threadIdx.x; k < n_pesos; k += THREADS) {
        const int j = k / c.saidas;
        const int i = k % c.saidas;
        c.w[k] -= taxa * (c.delta[i] * x[j]);
    }
    for (int i = threadIdx.x; i < c.saidas; i += THREADS) {
        c.b[i] -= taxa * c.delta[i];
    }
}

// Treino completo num único lançamento: <<<1, THREADS>>>.
__global__ void __launch_bounds__(THREADS)
k_treino(CamadaDev c1, CamadaDev c2, CamadaDev c3, const float* imagens,
         const int* rotulos, int n, int epocas, float taxa, double* perdas) {
    __shared__ float gradiente[Barenn::classes];
    for (int epoca = 0; epoca < epocas; ++epoca) {
        // Só a thread 0 acumula a perda da época, em double e na mesma
        // ordem da CPU (amostra por amostra).
        double perda_epoca = 0;
        for (int s = 0; s < n; ++s) {
            const float* x = imagens + s * Barenn::entradas;

            forward(c1, x);
            __syncthreads();
            forward(c2, c1.a);
            __syncthreads();
            forward(c3, c2.a);
            __syncthreads();

            // Perda (metade da soma dos erros quadráticos) e gradiente nos
            // escores: só 10 classes, feito por uma thread.
            if (threadIdx.x == 0) {
                const int rotulo = rotulos[s];
                float total = 0;
                for (int i = 0; i < Barenn::classes; ++i) {
                    float erro = c3.a[i] - (i == rotulo ? 1.0f : 0.0f);
                    total += 0.5f * erro * erro;
                    gradiente[i] = erro;
                }
                perda_epoca += total;
            }
            __syncthreads();

            // Backward da última para a primeira camada. Em cada camada:
            // delta, depois o gradiente da entrada (que lê os pesos antigos)
            // e só então a atualização dos pesos.
            calcula_delta(c3, gradiente);
            __syncthreads();
            grad_entrada(c3);
            __syncthreads();
            atualiza(c3, c2.a, taxa);
            calcula_delta(c2, c3.grad);  // independe da atualização de c3
            __syncthreads();
            grad_entrada(c2);
            __syncthreads();
            atualiza(c2, c1.a, taxa);
            calcula_delta(c1, c2.grad);  // independe da atualização de c2
            __syncthreads();
            // A entrada da rede não tem parâmetros: dispensa o grad de c1.
            atualiza(c1, x, taxa);
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            perdas[epoca] = perda_epoca;
        }
    }
}

// ---------------------------------------------------------------------------
// Alocação de uma camada no device a partir da camada da CPU.
// ---------------------------------------------------------------------------
CamadaDev carregar(Barenn::DenseLayer& cpu, bool final) {
    CamadaDev c{};
    c.entradas = cpu.entradas_camada();
    c.saidas = cpu.saidas_camada();
    c.final = final;
    const size_t n_pesos = static_cast<size_t>(c.entradas) * c.saidas;
    CUDA_CHECK(cudaMalloc(&c.w, n_pesos * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.b, c.saidas * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.z, c.saidas * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.a, c.saidas * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.delta, c.saidas * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c.grad, c.entradas * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(c.w, cpu.pesos().data(), n_pesos * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(c.b, cpu.vieses().data(), c.saidas * sizeof(float),
                          cudaMemcpyHostToDevice));
    return c;
}

// Copia os parâmetros treinados de volta para a camada da CPU.
void devolver(const CamadaDev& c, Barenn::DenseLayer& cpu) {
    CUDA_CHECK(cudaMemcpy(cpu.pesos().data(), c.w,
                          cpu.pesos().size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(cpu.vieses().data(), c.b,
                          cpu.vieses().size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

void liberar(CamadaDev& c) {
    for (float* p : {c.w, c.b, c.z, c.a, c.delta, c.grad}) {
        CUDA_CHECK(cudaFree(p));
    }
    c = CamadaDev{};
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        const auto inicio_total = chrono::steady_clock::now();
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        const chrono::duration<double> tempo_leitura =
            chrono::steady_clock::now() - inicio_total;
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Mesma rede e mesma inicialização da versão CPU: 64 -> 32 -> 16 -> 10.
        Barenn::DenseLayer l1(Barenn::entradas, 32);
        Barenn::DenseLayer l2(32, 16);
        Barenn::DenseLayer l3(16, Barenn::classes, true);
        array<double, epocas> perdas{};

        // Inicializa o contexto CUDA fora do cronômetro (custo fixo, ~centenas
        // de ms, que não depende do treino).
        CUDA_CHECK(cudaFree(nullptr));

        // A medição inclui as cópias Host->Device dos dados e dos pesos, o
        // kernel de treino e a cópia Device->Host dos pesos treinados.
        const auto inicio = chrono::steady_clock::now();

        // Conjunto de treino achatado: imagens contíguas (n x 64) e rótulos.
        const int n = static_cast<int>(treino.size());
        vector<float> imagens(static_cast<size_t>(n) * Barenn::entradas);
        vector<int> rotulos(n);
        for (int s = 0; s < n; ++s) {
            copy(treino[s].imagem.begin(), treino[s].imagem.end(),
                 imagens.begin() + static_cast<size_t>(s) * Barenn::entradas);
            rotulos[s] = treino[s].rotulo;
        }
        float* d_imagens = nullptr;
        int* d_rotulos = nullptr;
        double* d_perdas = nullptr;
        CUDA_CHECK(cudaMalloc(&d_imagens, imagens.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_rotulos, rotulos.size() * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_perdas, epocas * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_imagens, imagens.data(),
                              imagens.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rotulos, rotulos.data(),
                              rotulos.size() * sizeof(int),
                              cudaMemcpyHostToDevice));

        CamadaDev c1 = carregar(l1, false);
        CamadaDev c2 = carregar(l2, false);
        CamadaDev c3 = carregar(l3, true);

        k_treino<<<1, THREADS>>>(c1, c2, c3, d_imagens, d_rotulos, n, epocas,
                                 taxa, d_perdas);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(perdas.data(), d_perdas, epocas * sizeof(double),
                              cudaMemcpyDeviceToHost));
        devolver(c1, l1);
        devolver(c2, l2);
        devolver(c3, l3);
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        liberar(c1);
        liberar(c2);
        liberar(c3);
        CUDA_CHECK(cudaFree(d_imagens));
        CUDA_CHECK(cudaFree(d_rotulos));
        CUDA_CHECK(cudaFree(d_perdas));

        for (double& p : perdas) {
            p /= n;
        }

        // Mesma saída da versão CPU.
        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        cout << "Tempo de leitura " << tempo_leitura.count() << " s\n";
        Barenn::avaliar(l1, l2, l3, teste);
        const chrono::duration<double> tempo_total =
            chrono::steady_clock::now() - inicio_total;
        cout << "Tempo total " << tempo_total.count() << " s\n";
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
