# Barenn: relatório de otimização (CPU → CUDA)

Sumário executivo da APS2. Cada degrau de desempenho é um **bloco lógico**: Issue (diagnóstico, hipótese, métrica base) → branch → Pull Request → tag anotada → fechamento da Issue com a discussão. As saídas completas das medições estão em [`resultados/`](resultados/).

**Critério de corretude:** todas as versões produzem saída **bit a bit idêntica** à do código legado (`make validate` / `make validate-cuda`, comparando com [`ref_output.txt`](ref_output.txt)): acurácia 95,5% (1717/1797), perda média de teste 0,0432.

## Tabela consolidada

Tempo de treino (51 épocas, forward + perda + backward), média de 5 a 10 execuções.

| Versão | Tag | Notebook | Franky | Speedup Notebook | Speedup Franky |
|:--|:--|--:|--:|--:|--:|
| **Baseline Legado** | [`v0.0-baseline`](https://github.com/luizMM/awful-barenn/tree/v0.0-baseline) | 0,866 s | 0,471 s | 1,0× | 1,0× |
| **CPU Otimizada** | [`v0.4-openmp-leitura`](https://github.com/luizMM/awful-barenn/tree/v0.4-openmp-leitura) | 0,182 s | 0,130 s | **4,8×** | **3,6×** |
| **CUDA Inicial** | [`v0.5-cuda-inicial`](https://github.com/luizMM/awful-barenn/tree/v0.5-cuda-inicial) | 4,19 s | 3,92 s | 0,21× | 0,12× |
| **CUDA Otimizado** | [`v0.7-cuda-shared-unroll`](https://github.com/luizMM/awful-barenn/tree/v0.7-cuda-shared-unroll) | 0,545 s | 0,526 s | 1,6× | 0,90× |

## Todas as versões

| # | Bloco lógico | Issue | PR | Tag | Notebook | Franky |
|:--|:--|:--|:--|:--|--:|--:|
| 1 | Baseline reprodutível e profiling inicial | [#1](https://github.com/luizMM/awful-barenn/issues/1) | [#2](https://github.com/luizMM/awful-barenn/pull/2) | [`v0.0-baseline`](https://github.com/luizMM/awful-barenn/tree/v0.0-baseline) | 0,866 s | 0,471 s |
| 2 | Remove o `grad_in` descartado da camada 1 | [#3](https://github.com/luizMM/awful-barenn/issues/3) | [#4](https://github.com/luizMM/awful-barenn/pull/4) | [`v0.1-grad-in-morto`](https://github.com/luizMM/awful-barenn/tree/v0.1-grad-in-morto) | 0,719 s | 0,367 s |
| 3 | Pesos contíguos e buffers pré-alocados | [#5](https://github.com/luizMM/awful-barenn/issues/5) | [#6](https://github.com/luizMM/awful-barenn/pull/6) | [`v0.2-layout-contiguo`](https://github.com/luizMM/awful-barenn/tree/v0.2-layout-contiguo) | 0,642 s | 0,424 s ¹ |
| 4 | SIMD AVX2 + pesos transpostos | [#7](https://github.com/luizMM/awful-barenn/issues/7) | [#8](https://github.com/luizMM/awful-barenn/pull/8) | [`v0.3-simd-avx2`](https://github.com/luizMM/awful-barenn/tree/v0.3-simd-avx2) | 0,184 s | 0,133 s |
| 5 | OpenMP na leitura e Lei de Amdahl | [#9](https://github.com/luizMM/awful-barenn/issues/9) | [#10](https://github.com/luizMM/awful-barenn/pull/10) | [`v0.4-openmp-leitura`](https://github.com/luizMM/awful-barenn/tree/v0.4-openmp-leitura) | 0,182 s ² | 0,130 s ² |
| 6 | Porting inicial para CUDA | [#11](https://github.com/luizMM/awful-barenn/issues/11) | [#12](https://github.com/luizMM/awful-barenn/pull/12) | [`v0.5-cuda-inicial`](https://github.com/luizMM/awful-barenn/tree/v0.5-cuda-inicial) | 4,19 s | 3,92 s |
| 7 | Kernel persistente (1 lançamento) | [#13](https://github.com/luizMM/awful-barenn/issues/13) | [#14](https://github.com/luizMM/awful-barenn/pull/14) | [`v0.6-kernel-persistente`](https://github.com/luizMM/awful-barenn/tree/v0.6-kernel-persistente) | 1,09 s | 0,96 s |
| 8 | Shared memory + dimensões em tempo de compilação | [#15](https://github.com/luizMM/awful-barenn/issues/15) | [#16](https://github.com/luizMM/awful-barenn/pull/16) | [`v0.7-cuda-shared-unroll`](https://github.com/luizMM/awful-barenn/tree/v0.7-cuda-shared-unroll) | 0,545 s | 0,526 s |

¹ Regressão no Franky, investigada na Issue #5: sensibilidade ao layout de memória (deslocar um buffer em 32 bytes leva a 0,334 s). Ela desaparece na v0.3.
² O ganho da v0.4 é no tempo **total** (leitura em paralelo): 0,218 → 0,189 s no notebook e 0,152 → 0,135 s no Franky, com 8 threads, próximo do limite de Amdahl (1,19× e 1,17×). O treino é sequencial.

## Conclusões

- **CPU (4,8× no notebook e 3,6× no Franky):** o ganho veio de remover trabalho inútil, eliminar alocações e, principalmente, de **SIMD sem reordenar somas** (pesos transpostos, Issue #7). O treino não paraleliza entre núcleos: o SGD por amostra é sequencial e cada amostra leva ~1 µs, menos que o custo de uma região OpenMP (Issue #9).
- **GPU:** o porting direto ficou 23–30× mais lento que a CPU por causa de 2,34 milhões de lançamentos de kernel (Issue #11). O kernel persistente reduziu para 1 lançamento (Issue #13), e shared memory + laços desenrolados encurtaram o caminho crítico serial de cada amostra (Issue #15), num total de **7,5–7,7× sobre a CUDA inicial**.
- **Por que a GPU não supera a CPU:** com SGD por amostra só um bloco (1 dos 16/18 SMs) pode trabalhar, e o caminho crítico é uma cadeia serial de somas em ordem fixa (exigida pela saída bit a bit). O Nsight Compute mostra 61% dos ciclos em espera de barreira e throughput de SM < 2%. Superar a CPU exigiria paralelismo entre amostras (mini-batch), o que muda o algoritmo e foi deliberadamente não feito, para preservar o SGD exato.

## Ambientes de teste

| | Notebook | Franky (cluster) |
|:--|:--|:--|
| CPU | AMD Ryzen 7 5800H (Zen 3, 8C/16T) | Intel i7-13700T (nós CPU, `sunny_cpu`), i9-14900 (nós GPU) |
| GPU | NVIDIA GeForce RTX 3050 Laptop (Ampere, 16 SMs, 4 GB) | NVIDIA RTX A1000 (Ampere, 18 SMs, 8 GB), `sunny_gpu` |
| Compiladores | g++ 13.3, CUDA 12.0 | g++ 14.2, CUDA 12.8 |
| Flags CPU | `-O3 -march=x86-64-v3 -ffp-contract=off -fopenmp` (a partir da v0.3) | idem |
| Flags GPU | `-O3 -arch=sm_86 --fmad=false` | idem |
| Profilers | callgrind/cachegrind, Nsight Systems, Nsight Compute (com `sudo`) | Nsight Systems (contadores da GPU restritos) |

## Como reproduzir

```bash
git checkout <tag>
make validate && make bench        # CPU
make validate-cuda && make bench-cuda && make nsys   # GPU (a partir da v0.5)
make callgrind                     # profiling da CPU
./franky_submit.sh                 # Franky, CPU (no nó de login)
./franky_submit_gpu.sh             # Franky, GPU (a partir da v0.5)
```

---

# Enunciado original

## Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
