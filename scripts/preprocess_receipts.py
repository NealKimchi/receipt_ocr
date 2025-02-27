"""
Receipt OCR Preprocessing Script

This script processes receipt images for OCR model training by cleaning,
enhancing, and augmenting them. It can be run from the command line.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from datasets import load_dataset
import random
import io
import sys
import logging
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from objects.receipt_preprocessor import ReceiptPreprocessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("receipt_processing.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process receipt images for OCR training')
    
    parser.add_argument('--dataset_name', type=str, default='mychen76/invoices-and-receipts_ocr_v2',
                      help='Name of Hugging Face dataset or path to local dataset')
    
    parser.add_argument('--output_dir', type=str, default='preprocessed_receipts',
                      help='Directory to save processed images')
    
    parser.add_argument('--train_samples', type=int, default=None,
                      help='Number of training samples to process (None for all)')
    
    parser.add_argument('--valid_samples', type=int, default=None,
                      help='Number of validation samples to process (None for all)')
    
    parser.add_argument('--augment', action='store_true', default=True,
                      help='Apply data augmentation to training samples')
    
    parser.add_argument('--visualize', action='store_true',
                      help='Visualize sample processing results')
    
    parser.add_argument('--visualize_sample', type=int, default=0,
                      help='Sample index to visualize')
    
    return parser.parse_args()


def main():
    """Main function to run the preprocessing pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set the project paths
    current_dir = os.getcwd()
    project_root = os.path.dirname(current_dir)
    output_dir = os.path.join(project_root, args.output_dir)
    
    # Load the dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    try:
        # Try to load from Hugging Face
        dataset = load_dataset(args.dataset_name)
        logger.info(f"Successfully loaded dataset from Hugging Face: {args.dataset_name}")
    except:
        # If that fails, try to load as a local path
        logger.info(f"Could not load from Hugging Face, trying local path: {args.dataset_name}")
        try:
            dataset = load_dataset("imagefolder", data_dir=args.dataset_name)
            logger.info(f"Successfully loaded local dataset from: {args.dataset_name}")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            sys.exit(1)
    
    # Initialize the preprocessor
    preprocessor = ReceiptPreprocessor(dataset=dataset, output_dir=output_dir)
    
    # Process a small batch for visualization if requested
    if args.visualize:
        logger.info(f"Visualizing sample at index: {args.visualize_sample}")
        preprocessor.visualize_processed_image(
            sample_idx=args.visualize_sample,
            split="train",
            show_boxes=True
        )
    
    # Process training data
    logger.info("Processing training dataset...")
    train_stats = preprocessor.prepare_dataset(
        split="train",
        num_samples=args.train_samples,
        augment=args.augment
    )
    
    # Process validation data
    logger.info("Processing validation dataset...")
    val_stats = preprocessor.prepare_dataset(
        split="valid",
        num_samples=args.valid_samples,
        augment=False
    )
    
    # Print statistics
    logger.info("\nDataset processing results:")
    logger.info(f"Training images processed: {train_stats['total_processed']}")
    logger.info(f"Training text regions detected: {train_stats['total_boxes_detected']}")
    if train_stats['total_processed'] > 0:
        text_percentage = train_stats['images_with_boxes']/train_stats['total_processed']*100
        logger.info(f"Training images with text: {train_stats['images_with_boxes']} ({text_percentage:.1f}%)")
    
    logger.info(f"Validation images processed: {val_stats['total_processed']}")
    logger.info(f"Validation text regions detected: {val_stats['total_boxes_detected']}")
    if val_stats['total_processed'] > 0:
        text_percentage = val_stats['images_with_boxes']/val_stats['total_processed']*100
        logger.info(f"Validation images with text: {val_stats['images_with_boxes']} ({text_percentage:.1f}%)")
    
    logger.info(f"Preprocessed data saved to: {output_dir}")


if __name__ == "__main__":
    main()