#!/bin/bash
#SBATCH --job-name=receipt_ocr
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G  # Request more memory
#SBATCH --output=train_output_%j.log

# Load necessary modules
module purge
module load python3/3.11.4  # Use the version that worked for you

# Activate virtual environment if you have one
# source ~/ocr_env/bin/activate

# Run the training script with lower batch size
cd /path/to/your/receipt_ocr
python3 train/train_detection.py