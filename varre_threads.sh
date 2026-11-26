#!/usr/bin/env bash
# Varre o tamanho do bloco do kernel persistente (THREADS) e mede o treino.
# Uso: ./varre_threads.sh [N] [so-compilar|so-rodar]   (compilar requer nvcc)
# No Franky: "so-compilar" no nó de login e "so-rodar" no job de GPU.
set -euo pipefail
export LC_ALL=C
cd "$(dirname "$0")"
N=${1:-5}
MODO=${2:-tudo}
mkdir -p profiling
for T in 32 64 128 256 512 1024; do
    if [ "$MODO" != so-rodar ]; then
        nvcc -O3 -arch=sm_86 --fmad=false -std=c++17 -Xcompiler -fopenmp,-O3 \
            -DTHREADS="$T" barenn_cuda.cu -o "profiling/barenn_cuda_t$T"
    fi
    [ "$MODO" = so-compilar ] && continue
    if ./validate.sh "profiling/barenn_cuda_t$T" > /dev/null; then ok=OK; else ok=FALHA; fi
    echo "THREADS=$T validacao=$ok $(./bench.sh "profiling/barenn_cuda_t$T" "$N" | grep 'Tempo de treino')"
done
