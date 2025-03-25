#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --gres=gpu:1
#SBATCH --time=60
#SBATCH --job-name=merge-script
#SBATCH --output=/w/284/murdock/merge/outs/output/slurm-%j.out
#SBATCH --error=/w/284/murdock/merge/outs/error/slurm-%j.err

cd /w/284/murdock/merge/zipit

srun python zipit2.py