"""
Annotated Data Loader for Receipt OCR

This module provides dataset classes for loading receipt images with their annotations
for OCR model training.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import sys
# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from objects.annotations_loader import AnnotationsLoader


class ReceiptOCRDataset(Dataset):
    """Dataset for training a text recognition model using annotated receipt data."""
    
    def __init__(self, root_dir, annotations_file=None, split='train', transform=None, 
                 line_height_threshold=5, min_line_length=3, preprocessed_data_dir=None):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory of receipt images
            annotations_file: Path to annotations JSON file
            split: 'train' or 'valid'
            transform: Transforms to apply to images
            line_height_threshold: Maximum y-distance for boxes to be in the same line
            min_line_length: Minimum number of characters in a line to include it
            preprocessed_data_dir: Directory to save/load preprocessed data
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.line_height_threshold = line_height_threshold
        self.min_line_length = min_line_length
        self.preprocessed_data_dir = preprocessed_data_dir
        
        # Load annotations
        self.loader = AnnotationsLoader(annotations_file)
        self.image_files = [f for f in os.listdir(self.root_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        self.samples = []
        if preprocessed_data_dir:
            os.makedirs(preprocessed_data_dir, exist_ok=True)
            preprocessed_file = os.path.join(preprocessed_data_dir, f"{split}_samples.json")
            
            if os.path.exists(preprocessed_file):
                # Load preprocessed data
                with open(preprocessed_file, 'r') as f:
                    self.samples = json.load(f)
                print(f"Loaded {len(self.samples)} preprocessed samples from {preprocessed_file}")
            else:
                # Process raw data
                self._process_raw_data()
                
                # Save preprocessed data
                with open(preprocessed_file, 'w') as f:
                    json.dump(self.samples, f)
                print(f"Saved {len(self.samples)} preprocessed samples to {preprocessed_file}")
        else:
            self._process_raw_data()
        
        # Create character map
        self.create_char_map()
        
    def _process_raw_data(self):
        """Process raw data to create samples."""
        print(f"Processing raw data from {self.root_dir}...")
        
        for img_file in self.image_files:
            img_path = os.path.join(self.root_dir, img_file)
            
            if img_file in self.loader.annotations:
                ocr_data = self.loader.annotations[img_file]
                
                text_boxes = self.loader.parse_ocr_annotations(ocr_data)
                
                if text_boxes:
                    lines = self.loader.convert_boxes_to_lines(text_boxes, self.line_height_threshold)
                    
                    for i, line in enumerate(lines):
                        if len(line) >= self.min_line_length:
                            line_text = ' '.join(text for text, _ in line)
                            
                            boxes = [box for _, box in line]
                            min_x = max(0, int(min(box[0] for box in boxes)) - 2)
                            min_y = max(0, int(min(box[1] for box in boxes)) - 2)
                            max_x = int(max(box[0] + box[2] for box in boxes)) + 2
                            max_y = int(max(box[1] + box[3] for box in boxes)) + 2
                            
                            line_box = [min_x, min_y, max_x - min_x, max_y - min_y]
                            
                            # Add sample
                            self.samples.append({
                                'image_path': img_path,
                                'line_box': line_box,
                                'text': line_text
                            })
        
        print(f"Processed {len(self.samples)} text line samples")
        
    def create_char_map(self):
        """Create mapping between characters and indices."""
        # Get all unique characters from the dataset
        self.chars = set()
        for sample in self.samples:
            text = sample['text']
            self.chars.update(list(text))
        
        # Add special tokens
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        for token in special_tokens:
            self.chars.add(token)
            
        # Create char-to-index and index-to-char mappings
        self.char_to_idx = {char: i for i, char in enumerate(sorted(list(self.chars)))}
        self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}
        self.num_classes = len(self.char_to_idx)
        
        print(f"Created character map with {self.num_classes} characters")
        
    def encode_text(self, text):
        """Convert text string to tensor of indices."""
        indices = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in text]
        return torch.tensor(indices, dtype=torch.long)
    
    def decode_text(self, indices):
        """Convert tensor of indices to text string."""
        return ''.join([self.idx_to_char[i.item()] for i in indices if i.item() != self.char_to_idx['<pad>']])
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample['image_path']
        line_box = sample['line_box']
        text = sample['text']
        
        try:
            # Load the image
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Could not load image: {img_path}")
                
            # Extract the line region
            x, y, w, h = line_box
            line_region = image[y:y+h, x:x+w]
            
            # Convert to grayscale
            if len(line_region.shape) == 3:
                line_region = cv2.cvtColor(line_region, cv2.COLOR_BGR2GRAY)
                
            # Convert to PIL Image
            line_region = Image.fromarray(line_region)
            
        except Exception as e:
            print(f"Error loading image region {img_path}: {e}")
            # Return a blank image as fallback
            line_region = Image.new('L', (100, 32), color=255)
            
        # Apply transforms if any
        if self.transform:
            line_region = self.transform(line_region)
            
        # Encode the text
        target = self.encode_text(text)
        
        return line_region, target, text


def get_receipt_ocr_dataloaders(root_dir, annotations_file, batch_size=32, train_transform=None, 
                              val_transform=None, preprocessed_data_dir=None):
    """Create DataLoaders for the receipt OCR task."""
    
    # Create datasets
    train_dataset = ReceiptOCRDataset(
        root_dir=root_dir,
        annotations_file=annotations_file,
        split='train',
        transform=train_transform,
        preprocessed_data_dir=preprocessed_data_dir
    )
    
    val_dataset = ReceiptOCRDataset(
        root_dir=root_dir,
        annotations_file=annotations_file,
        split='valid',
        transform=val_transform,
        preprocessed_data_dir=preprocessed_data_dir
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_text_batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_text_batch
    )
    
    return train_loader, val_loader, train_dataset, val_dataset


def collate_text_batch(batch):
    """Custom collate function for variable length text recognition batches."""
    images, targets, raw_texts = zip(*batch)
    
    # Stack images
    images = torch.stack(images)
    return images, targets, raw_texts


def convert_huggingface_annotations(dataset, output_file):
    annotations = {}
    
    for example in dataset:
        # Get image file name
        if 'image_path' in example:
            img_file = os.path.basename(example['image_path'])
        elif 'image' in example:
            img_file = f"image_{example['id'] if 'id' in example else len(annotations)}.jpg"
        else:
            continue
            
        ocr_data = {}
        
        if 'ocr_words' in example:
            ocr_data['ocr_words'] = example['ocr_words']
        elif 'words' in example:
            ocr_data['ocr_words'] = json.dumps(example['words'])
        if 'ocr_boxes' in example:
            ocr_data['ocr_boxes'] = example['ocr_boxes']
        elif 'boxes' in example:
            ocr_data['ocr_boxes'] = json.dumps(example['boxes'])
        if 'parsed_data' in example and example['parsed_data']:
            ocr_data['parsed_data'] = example['parsed_data']
        if 'raw_data' in example and example['raw_data']:
            ocr_data['raw_data'] = example['raw_data']
        if ocr_data:
            annotations[img_file] = ocr_data
    
    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2)
        
    print(f"Converted {len(annotations)} annotations to {output_file}")