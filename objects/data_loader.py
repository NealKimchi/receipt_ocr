"""
Receipt OCR Data Loading Module with Line Grouping

This module handles loading and processing receipt images with bounding box coordinates,
and groups them into text lines for OCR training.
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from collections import defaultdict


class ReceiptLineDataset(Dataset):
    """Dataset for training a text recognition model on receipt text lines."""
    
    def __init__(self, root_dir, split='train', transform=None, max_text_length=50, 
                annotations_file=None, line_height_threshold=10, min_line_boxes=3):
        """
        Args:
            root_dir: Root directory of preprocessed data
            split: 'train' or 'valid'
            transform: Optional transforms to apply to images
            max_text_length: Maximum length of text for padding
            annotations_file: Path to a file containing text annotations for the lines
            line_height_threshold: Maximum y-distance between boxes to be considered part of the same line
            min_line_boxes: Minimum number of boxes to form a valid line
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.max_text_length = max_text_length
        self.line_height_threshold = line_height_threshold
        self.min_line_boxes = min_line_boxes
        
        # Check if the directory exists
        if not os.path.exists(self.root_dir):
            raise ValueError(f"Directory not found: {self.root_dir}")
        
        # Get all image files from the directory
        self.image_files = [f for f in os.listdir(self.root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Load text annotations if provided
        self.annotations = {}
        if annotations_file and os.path.exists(annotations_file):
            with open(annotations_file, 'r', encoding='utf-8') as f:
                self.annotations = json.load(f)
        
        # Create a list of (image_path, bbox_path) tuples
        self.samples = []
        self.all_lines = []  # Will store (image_index, line_boxes) tuples
        
        for i, img_file in enumerate(self.image_files):
            img_path = os.path.join(self.root_dir, img_file)
            # Look for corresponding text file with bbox coordinates
            base_name = os.path.splitext(img_file)[0]
            txt_file = f"{base_name}.txt"
            txt_path = os.path.join(self.root_dir, txt_file)
            
            if os.path.exists(txt_path):
                bboxes = self.parse_bbox_file(txt_path)
                lines = self.group_boxes_into_lines(bboxes)
                
                for line_boxes in lines:
                    if len(line_boxes) >= self.min_line_boxes:
                        # Use text annotation if available, otherwise empty
                        text = self.annotations.get(f"{img_file}_{len(self.all_lines)}", "")
                        self.samples.append((img_path, line_boxes, text))
                        self.all_lines.append((i, line_boxes))
        
        # Create character map from all available text
        self.create_char_map()
        
    def create_char_map(self):
        """Create mapping between characters and indices."""
        # Create a basic character set for receipt text
        self.chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,;:!?-_'\"()[]$%&*/+= ")
        
        # Add any text from annotations
        for _, _, text in self.samples:
            if text:
                self.chars.update(list(text))
        
        # Add special tokens
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        for token in special_tokens:
            self.chars.add(token)
            
        # Create char-to-index and index-to-char mappings
        self.char_to_idx = {char: i for i, char in enumerate(sorted(self.chars))}
        self.idx_to_char = {i: char for char, i in self.char_to_idx.items()}
        self.num_classes = len(self.char_to_idx)
        
    def parse_bbox_file(self, bbox_path):
        """Parse a file containing bounding box coordinates in x,y,w,h format."""
        bboxes = []
        try:
            with open(bbox_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    # Parse as comma-separated format
                    parts = line.strip().split(',')
                    if len(parts) >= 4:  # x,y,w,h format
                        try:
                            x, y, w, h = map(float, parts[:4])
                            # Skip invalid boxes
                            if w > 0 and h > 0:
                                bboxes.append((x, y, w, h))
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error reading bbox file {bbox_path}: {e}")
        
        return bboxes
    
    def group_boxes_into_lines(self, bboxes):
        """Group bounding boxes into text lines based on y-coordinate proximity."""
        if not bboxes:
            return []
            
        # Sort boxes by y-coordinate
        sorted_boxes = sorted(bboxes, key=lambda box: box[1])
        
        # Group boxes into lines
        lines = []
        current_line = [sorted_boxes[0]]
        current_y = sorted_boxes[0][1]
        
        for box in sorted_boxes[1:]:
            box_y = box[1]
            # If y-coordinate is close to current line's y, add to current line
            if abs(box_y - current_y) <= self.line_height_threshold:
                current_line.append(box)
            else:
                # Otherwise, start a new line
                if current_line:
                    lines.append(current_line)
                current_line = [box]
                current_y = box_y
                
        # Add the last line if not empty
        if current_line:
            lines.append(current_line)
            
        # Sort boxes within each line by x-coordinate
        for i in range(len(lines)):
            lines[i] = sorted(lines[i], key=lambda box: box[0])
            
        return lines
    
    def get_line_region(self, image, line_boxes):
        """Extract a text line region from the image using the bounding boxes."""
        if not line_boxes:
            return None
            
        # Calculate the bounding box that encompasses all boxes in the line
        min_x = min(box[0] for box in line_boxes)
        min_y = min(box[1] for box in line_boxes)
        max_x = max(box[0] + box[2] for box in line_boxes)
        max_y = max(box[1] + box[3] for box in line_boxes)
        
        # Add a small margin
        margin = 2
        min_x = max(0, min_x - margin)
        min_y = max(0, min_y - margin)
        max_x = max_x + margin
        max_y = max_y + margin
        
        # Ensure coordinates are within image boundaries
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]
            
        min_x = min(min_x, width - 1)
        min_y = min(min_y, height - 1)
        max_x = min(max_x, width)
        max_y = min(max_y, height)
        
        # Extract the region
        if isinstance(image, Image.Image):
            region = image.crop((min_x, min_y, max_x, max_y))
        else:
            region = image[int(min_y):int(max_y), int(min_x):int(max_x)]
            if not isinstance(region, Image.Image):
                region = Image.fromarray(region)
                
        return region
    
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
        img_path, line_boxes, text = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('L', (100, 32), color=255)
            line_boxes = []
            
        original_image = image
        line_region = self.get_line_region(image, line_boxes)
        
        if line_region is None:
            line_region = Image.new('L', (100, 32), color=255)
        if self.transform:
            line_region = self.transform(line_region)
        if not text:
            text = f"line_{idx}"
        target = self.encode_text(text)
        
        # Return the line region, target text, raw text, original image, and line boxes
        return line_region, target, text, line_boxes


def get_text_line_dataloaders(root_dir, batch_size=32, train_transform=None, val_transform=None,
                           annotations_file=None, line_height_threshold=15, min_line_boxes=3):
    """Create DataLoaders for the text line recognition task."""
    
    # Create datasets
    train_dataset = ReceiptLineDataset(
        root_dir=root_dir,
        split='train',
        transform=train_transform,
        annotations_file=annotations_file,
        line_height_threshold=line_height_threshold,
        min_line_boxes=min_line_boxes
    )
    
    val_dataset = ReceiptLineDataset(
        root_dir=root_dir,
        split='valid',
        transform=val_transform,
        annotations_file=annotations_file,
        line_height_threshold=line_height_threshold,
        min_line_boxes=min_line_boxes
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
    line_regions, targets, raw_texts, line_boxes = zip(*batch)
    
    line_regions = torch.stack(line_regions)
    return line_regions, targets, raw_texts, line_boxes