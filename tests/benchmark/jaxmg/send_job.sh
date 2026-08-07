#!/bin/bash -l
#SBATCH --account=neuquass
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4            # This is gpu per node
#SBATCH --ntasks-per-node=4     # Put this to one if n_nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=%j.out

module --force purge
module load Stages/2026
module load GCCcore/.14.3.0
module load CUDA/13
module load Python/3.13.5
module load cuDNN/9.19.0.56-CUDA-13
module load NCCL/default-CUDA-13

source "$SCRATCH"/env/jVMC/bin/activate

export JVMC_USE_DISTRIBUTED=true

srun python -u minsr.py
