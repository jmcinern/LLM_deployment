#!/bin/bash
#SBATCH --job-name=QUANT_AWQ_Instruct
#SBATCH --output=./out/quant_awq_instruct_%j.out
#SBATCH --error=./err/quant_awq_instruct_%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=k2-gpu-v100  
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=josephmcinerney7575@gmail.com

module load python3/3.10.5/gcc-9.3.0 # available python
module load libs/nvidia-cuda/12.4.0/bin # cuda
source /mnt/scratch2/users/40460549/cpt-dail/awq_env/bin/activate
cd $SLURM_SUBMIT_DIR

# run it
python quantize_awq.py

