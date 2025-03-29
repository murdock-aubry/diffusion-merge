#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --gres=gpu:t4:4
#SBATCH --qos=m3
#SBATCH --time=1:59:00
#SBATCH -c 30
#SBATCH --mem=150G
#SBATCH --output=/scratch/ssd004/scratch/murdock/diffusion-merge/outs/output/slurm-%j.out
#SBATCH --error=/scratch/ssd004/scratch/murdock/diffusion-merge/outs/error/slurm-%j.err

cd /scratch/ssd004/scratch/murdock/diffusion-merge/benchmark

srun python bench-script.py