#!/bin/bash
#SBATCH --partition=gpunodes
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=60
#SBATCH --job-name=vae-train
#SBATCH --output=/w/284/murdock/merge/outs/output/slurm-%j.out
#SBATCH --error=/w/284/murdock/merge/outs/error/slurm-%j.err

cd /w/284/murdock/merge
MODEL_NAME="sd1.4-cocotuned"
LATENT="2457"
MODEL="dim${LATENT}_epoch100_time_conditioned.pth"

python vae/test.py --model_path "/w/383/murdock/models/vae/$MODEL_NAME/$MODEL" \
                  --data_path "/w/383/murdock/hidden_reps/$MODEL_NAME/representations_50.pt" \
                  --output_path "/w/383/murdock/encoded_reps/$MODEL_NAME/encoded_data_${LATENT}.pt" \
                  --batch_size 32