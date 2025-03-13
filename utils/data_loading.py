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
        
        # Define transforms for images only - no bounding boxes
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
        
        print(f"Sample {idx} keys: {list(sample.keys())}")
        if 'ocr_boxes' in sample:
            print(f"ocr_boxes type: {type(sample['ocr_boxes'])}")
            if isinstance(sample['ocr_boxes'], str):
                print(f"ocr_boxes length: {len(sample['ocr_boxes'])}")
                print(f"ocr_boxes preview: {sample['ocr_boxes'][:100]}")  # Show first 100 chars
            elif isinstance(sample['ocr_boxes'], list):
                print(f"ocr_boxes length: {len(sample['ocr_boxes'])}")
                if len(sample['ocr_boxes']) > 0:
                    print(f"First box: {sample['ocr_boxes'][0]}")
        
        # Load image 
        image = self._load_image(sample)
        original_h, original_w = image.shape[:2]
        
        # Extract OCR boxes
        boxes = []
        if 'ocr_boxes' in sample:
            ocr_boxes = sample['ocr_boxes']
            
            # Convert from string if needed
            if isinstance(ocr_boxes, str):
                try:
                    ocr_boxes = json.loads(ocr_boxes)
                except:
                    ocr_boxes = []
            
            # Process each box
            for box_data in ocr_boxes:
                try:
                    # Format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], (text, confidence)
                    polygon = box_data[0]
                    text_conf = box_data[1]
                    
                    # Extract coordinates
                    x_vals = [p[0] for p in polygon]
                    y_vals = [p[1] for p in polygon]
                    x1, y1 = min(x_vals), min(y_vals)
                    x2, y2 = max(x_vals), max(y_vals)
                    
                    # Extract confidence
                    conf = float(text_conf[1]) if isinstance(text_conf, (list, tuple)) and len(text_conf) > 1 else 1.0
                    
                    # Create normalized box [conf, x1, y1, x2, y2]
                    norm_x1 = float(x1) / original_w
                    norm_y1 = float(y1) / original_h
                    norm_x2 = float(x2) / original_w
                    norm_y2 = float(y2) / original_h
                    
                    # Add to boxes list
                    boxes.append([conf, norm_x1, norm_y1, norm_x2, norm_y2])
                except Exception as e:
                    print(f"Error processing box: {e}")
        
        # Create text map (for segmentation)
        text_map = np.zeros((original_h, original_w), dtype=np.float32)
        for box in boxes:
            _, x1, y1, x2, y2 = box
            # Convert normalized coordinates back to absolute
            abs_x1 = int(x1 * original_w)
            abs_y1 = int(y1 * original_h)
            abs_x2 = int(x2 * original_w)
            abs_y2 = int(y2 * original_h)
            # Fill the text map
            text_map[abs_y1:abs_y2, abs_x1:abs_x2] = 1.0
        
        # Apply transforms to image
        transformed = self.transforms(image=image, mask=text_map)
        image = transformed['image']  # Now a tensor
        text_map = transformed['mask']  # Also a tensor
        
        # Add batch dimension to text map
        text_map = text_map.unsqueeze(0)
        
        # Debug info
        print(f"Sample {idx}: {len(boxes)} boxes found")
        if len(boxes) > 0:
            print(f"First box: {boxes[0]}")
        
        return {
            'image': image,
            'text_map': text_map,
            'boxes': boxes,
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
    
    # Custom collate function
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        text_maps = torch.stack([item['text_map'] for item in batch])
        boxes = [item['boxes'] for item in batch]
        image_ids = [item['image_id'] for item in batch]
        
        return {
            'image': images,
            'text_map': text_maps,
            'boxes': boxes,
            'image_id': image_ids
        }
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader