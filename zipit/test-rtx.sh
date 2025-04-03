#!/bin/bash
#SBATCH --partition=rtx6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=120
#SBATCH --job-name=finetuning-script
#SBATCH --output=/scratch/ssd004/scratch/murdock/diffusion-merge/outs/output/slurm-%j.out
#SBATCH --error=/scratch/ssd004/scratch/murdock/diffusion-merge/outs/error/slurm-%j.err

cd /scratch/ssd004/scratch/murdock/diffusion-merge/zipit

srun python test.py