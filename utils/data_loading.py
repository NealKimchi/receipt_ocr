import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from datasets import load_dataset
import re

class ReceiptDataset(Dataset):
    """Dataset for receipt OCR text detection"""
    def __init__(self, dataset_name="mychen76/invoices-and-receipts_ocr_v2", split='train', 
                 image_size=(512, 512), max_samples=None, cache_dir=None):
        """Initialize dataset with Hugging Face dataset"""
        self.image_size = image_size
        
        # Load dataset from Hugging Face
        if split == 'val':
            # Use a portion of train as validation
            self.dataset = load_dataset(dataset_name, split='train', cache_dir=cache_dir)
            val_size = int(0.1 * len(self.dataset))
            splits = self.dataset.train_test_split(test_size=val_size, seed=42)
            self.dataset = splits['test']
        else:  # train
            self.dataset = load_dataset(dataset_name, split='train', cache_dir=cache_dir)
            val_size = int(0.1 * len(self.dataset))
            splits = self.dataset.train_test_split(test_size=val_size, seed=42)
            self.dataset = splits['train']
            
        # Limit number of samples if specified
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
        print(f"Loaded {len(self.dataset)} samples for {split} split")
        
        # Define transforms for images and masks
        self.transforms = A.Compose([
            A.Resize(height=image_size[0], width=image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get sample from dataset
        sample = self.dataset[idx]
        
        # Load image 
        image = self._load_image(sample)
        original_h, original_w = image.shape[:2]
        
        # Extract OCR boxes from raw_data
        words, boxes, confidences = self._parse_raw_data(sample)
        
        # Create text map based on boxes
        text_map = np.zeros((original_h, original_w), dtype=np.float32)
        valid_boxes = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Convert to absolute pixel coordinates
            x1_px, y1_px = int(x1), int(y1)
            x2_px, y2_px = int(x2), int(y2)
            
            # Ensure valid coordinates
            if x2_px <= x1_px or y2_px <= y1_px:
                continue
                
            x1_px = max(0, min(x1_px, original_w-1))
            y1_px = max(0, min(y1_px, original_h-1))
            x2_px = max(x1_px+1, min(x2_px, original_w))
            y2_px = max(y1_px+1, min(y2_px, original_h))
            
            # Add to valid boxes
            valid_boxes.append([x1_px, y1_px, x2_px, y2_px])
            
            # Fill text map
            text_map[y1_px:y2_px, x1_px:x2_px] = 1.0
        
        # Normalize valid boxes
        normalized_boxes = []
        for box in valid_boxes:
            x1, y1, x2, y2 = box
            norm_x1 = float(x1) / original_w
            norm_y1 = float(y1) / original_h
            norm_x2 = float(x2) / original_w
            norm_y2 = float(y2) / original_h
            normalized_boxes.append([norm_x1, norm_y1, norm_x2, norm_y2])
        
        # Apply transforms
        transformed = self.transforms(image=image, mask=text_map)
        image = transformed['image']
        text_map = transformed['mask']
        
        # Add batch dimension to text map
        text_map = text_map.unsqueeze(0)
        
        # Print debug info
        if idx % 100 == 0:  # Only print occasionally to reduce output
            print(f"Sample {idx}: Found {len(valid_boxes)} valid boxes")
        
        return {
            'image': image,
            'text_map': text_map,
            'boxes': normalized_boxes,
            'image_id': sample.get('id', str(idx))
        }
    
    def _load_image(self, sample):
        """Load image from sample - handles different formats"""
        try:
            if 'image' in sample and hasattr(sample['image'], 'convert'):
                # Image is already loaded
                pil_image = sample['image'].convert('RGB')
                return np.array(pil_image)
            
            elif 'image' in sample and isinstance(sample['image'], dict) and 'bytes' in sample['image']:
                # Load from bytes
                import io
                image_bytes = sample['image']['bytes']
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                return np.array(pil_image)
                
            else:
                # Return black image if no valid format found
                print(f"No valid image found in sample: {list(sample.keys())}")
                return np.zeros((256, 256, 3), dtype=np.uint8)
                
        except Exception as e:
            print(f"Error loading image: {e}")
            return np.zeros((256, 256, 3), dtype=np.uint8)
    
    def _parse_ocr_boxes(self, ocr_boxes_str):
        """Parse ocr_boxes string into structured data"""
        words = []
        boxes = []
        confidences = []
        
        # Extract each box using regex
        box_pattern = r"\[\[\[(.*?)\]\], \((.*?)\)\]"
        matches = re.findall(box_pattern, ocr_boxes_str)
        
        for coords_str, text_conf_str in matches:
            try:
                # Parse coordinates
                coords_pattern = r"(\d+\.\d+), (\d+\.\d+)"
                coord_matches = re.findall(coords_pattern, coords_str)
                
                if not coord_matches or len(coord_matches) < 3:
                    continue
                    
                # Extract points
                points = []
                for x_str, y_str in coord_matches:
                    points.append([float(x_str), float(y_str)])
                
                # Calculate bounding box
                x_vals = [p[0] for p in points]
                y_vals = [p[1] for p in points]
                x1, y1 = min(x_vals), min(y_vals)
                x2, y2 = max(x_vals), max(y_vals)
                
                # Parse text and confidence
                text_conf_pattern = r"'(.*?)', ([\d\.]+)"
                text_conf_match = re.search(text_conf_pattern, text_conf_str)
                
                if text_conf_match:
                    text = text_conf_match.group(1)
                    conf = float(text_conf_match.group(2))
                    
                    # Add to lists
                    words.append(text)
                    boxes.append([x1, y1, x2, y2])
                    confidences.append(conf)
            except Exception as e:
                # Silently ignore problematic boxes
                continue
        
        return words, boxes, confidences
    
    def _parse_raw_data(self, sample):
        """Parse raw_data field to extract words, boxes, and confidences"""
        words = []
        boxes = []
        confidences = []
        
        if 'raw_data' not in sample:
            return words, boxes, confidences
            
        try:
            raw_data = json.loads(sample['raw_data'])
            
            if 'ocr_boxes' in raw_data:
                ocr_boxes_str = raw_data['ocr_boxes']
                words, boxes, confidences = self._parse_ocr_boxes(ocr_boxes_str)
                        
            return words, boxes, confidences
            
        except Exception as e:
            # If there's an error, just return empty lists
            return words, boxes, confidences


def get_data_loaders(dataset_name="mychen76/invoices-and-receipts_ocr_v2", 
                     batch_size=8, image_size=(512, 512), num_workers=4, 
                     max_samples=None, cache_dir=None):
    """Create data loaders for training and validation"""
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = ReceiptDataset(
        dataset_name=dataset_name, 
        split='train', 
        image_size=image_size,
        max_samples=max_samples,
        cache_dir=cache_dir
    )
    
    print("Loading validation dataset...")
    val_dataset = ReceiptDataset(
        dataset_name=dataset_name, 
        split='val', 
        image_size=image_size,
        max_samples=max_samples,
        cache_dir=cache_dir
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=None  # Use default collate_fn as we're returning a dict
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=None  # Use default collate_fn as we're returning a dict
    )
    
    return train_loader, val_loader