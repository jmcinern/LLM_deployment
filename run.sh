#!/bin/bash
#SBATCH --job-name=QUANT_AWQ_Instruct
#SBATCH --output=./out/quant_awq_instruct_%j.out
#SBATCH --error=./err/quant_awq_instruct_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=k2-gpu-a100  
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josephmcinerney7575@gmail.com

# run it
python quantize_awq.py

