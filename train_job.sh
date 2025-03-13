#!/bin/bash
#SBATCH --job-name=receipt_ocr
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G         # Request much more memory
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=train_output_%j.log

# Load necessary modules
module purge
module load python3/3.11.4

# Run the training script
cd $SLURM_SUBMIT_DIR
python3 train/train_detection.py