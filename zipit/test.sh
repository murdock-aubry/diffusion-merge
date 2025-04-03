#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --gres=gpu:t4:2
#SBATCH --qos=m3
#SBATCH --time=1:59:00
#SBATCH -c 30
#SBATCH --mem=100G
#SBATCH --output=/scratch/ssd004/scratch/murdock/diffusion-merge/outs/output/slurm-%j.out
#SBATCH --error=/scratch/ssd004/scratch/murdock/diffusion-merge/outs/error/slurm-%j.err

cd /w/284/murdock/merge/zipit

srun python test.py