#!/bin/bash
#SBATCH --job-name=receipt_ocr
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G          # Request more memory (increase from 32G to 64G)
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# Load necessary modules
module purge
module load python3/3.11.4

# Run the training script with lower batch size
cd $SLURM_SUBMIT_DIR
python3 train/train_detection.py