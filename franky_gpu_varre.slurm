#!/bin/bash
# Varredura do tamanho de bloco da versão CUDA no Franky.
# Antes, no nó de login: module load cuda/12.8.1 && ./varre_threads.sh 5 so-compilar
#SBATCH --job-name=barenn-varre
#SBATCH --output=resultados/franky_varre_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --partition=sunny_gpu
#SBATCH --gres=gpu:1

echo "Host: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
./varre_threads.sh "${N:-5}" so-rodar
