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
import io
import requests
from tqdm import tqdm

class ReceiptDataset(Dataset):
    """Dataset for receipt OCR text detection using Hugging Face datasets"""
    def __init__(self, dataset_name="mychen76/invoices-and-receipts_ocr_v2", split='train', 
                 transform=None, image_size=(512, 512), max_samples=None, cache_dir=None):
        """
        Args:
            dataset_name (str): Hugging Face dataset name
            split (str): 'train', 'val', or 'test'
            transform (callable, optional): Optional transform to be applied on a sample
            image_size (tuple): Target image size
            max_samples (int, optional): Maximum number of samples to load (for debugging)
            cache_dir (str, optional): Directory to cache the dataset
        """
        self.image_size = image_size
        
        # Load dataset from Hugging Face
        if split == 'val':
            # Use a portion of train as validation
            self.dataset = load_dataset(dataset_name, split='train', cache_dir=cache_dir)
            # Split train into train and val
            val_size = int(0.1 * len(self.dataset))
            splits = self.dataset.train_test_split(test_size=val_size, seed=42)
            self.dataset = splits['test']
        elif split == 'test':
            # Use a portion of train as test
            self.dataset = load_dataset(dataset_name, split='train', cache_dir=cache_dir)
            # Split train into main and test
            test_size = int(0.1 * len(self.dataset))
            splits = self.dataset.train_test_split(test_size=test_size, seed=123)
            self.dataset = splits['test']
        else:  # train
            self.dataset = load_dataset(dataset_name, split='train', cache_dir=cache_dir)
            # Remove validation and test portions
            val_size = int(0.1 * len(self.dataset))
            test_size = int(0.1 * len(self.dataset))
            splits = self.dataset.train_test_split(test_size=val_size+test_size, seed=42)
            self.dataset = splits['train']
            
        # Limit number of samples if specified
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            
        print(f"Loaded {len(self.dataset)} samples for {split} split")
        
        # Set default transform if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(height=image_size[0], width=image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Load sample
        sample = self.dataset[idx]
        
        # Load image
        image = self._load_image_from_sample(sample)
        h, w = image.shape[:2]
        
        # Directly extract and process boxes from raw JSON
        boxes = []
        if 'ocr_boxes' in sample:
            try:
                ocr_boxes = sample['ocr_boxes']
                # Convert from string if needed
                if isinstance(ocr_boxes, str):
                    import json
                    ocr_boxes = json.loads(ocr_boxes)
                    
                # Process each box
                for box_data in ocr_boxes:
                    # Extract polygon points and confidence
                    polygon = box_data[0]  # [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    text_conf = box_data[1]  # (text, confidence)
                    
                    # Get box coordinates
                    x_coords = [p[0] for p in polygon]
                    y_coords = [p[1] for p in polygon]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    # Get confidence
                    conf = text_conf[1] if isinstance(text_conf, (list, tuple)) and len(text_conf) > 1 else 1.0
                    
                    # Create box in format [conf, x1, y1, x2, y2]
                    boxes.append([float(conf), float(x1), float(y1), float(x2), float(y2)])
            except Exception as e:
                print(f"Error processing OCR boxes: {e}")
        
        # Create text map
        text_map = np.zeros((h, w), dtype=np.float32)
        for box in boxes:
            _, x1, y1, x2, y2 = box
            # Convert to integers for indexing
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Make sure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x1 < x2 and y1 < y2:
                text_map[y1:y2, x1:x2] = 1.0
        
        # Apply transforms
        if self.transform:
            # For albumentations, we need to match our bbox format with what it expects
            transformed = self.transform(
                image=image,
                mask=text_map
            )
            image = transformed['image']
            text_map = transformed['mask']
        
        # Convert to tensors
        image = torch.as_tensor(image)
        text_map = torch.as_tensor(text_map, dtype=torch.float32).unsqueeze(0)
        
        # Print debug info
        print(f"Sample {idx}, Number of boxes: {len(boxes)}")
        if len(boxes) > 0:
            print(f"First box: {boxes[0]}")
        
        return {
            'image': image,
            'text_map': text_map,
            'boxes': boxes,
            'image_id': sample.get('id', str(idx))
        }

    def _process_ocr_boxes(self, sample, image_shape):
        """Process OCR boxes from the dataset.
        
        Args:
            sample: Dataset sample
            image_shape: Shape of the image (h, w)
            
        Returns:
            list: List of boxes in format [confidence, x1, y1, x2, y2]
        """
        h, w = image_shape
        processed_boxes = []
        
        # Try to extract ocr_boxes from the sample
        if 'ocr_boxes' in sample:
            raw_boxes = sample['ocr_boxes']
            
            # Handle different possible formats
            if isinstance(raw_boxes, str):
                # Try to parse JSON string
                try:
                    import json
                    raw_boxes = json.loads(raw_boxes)
                except:
                    # If parsing fails, return empty list
                    return []
            
            # Process the boxes based on the format in your dataset
            for box_data in raw_boxes:
                try:
                    # Format from the example: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], (text, confidence)
                    polygon_points = box_data[0]
                    text_conf_tuple = box_data[1]
                    
                    # Extract polygon points
                    x1 = min(point[0] for point in polygon_points)
                    y1 = min(point[1] for point in polygon_points)
                    x2 = max(point[0] for point in polygon_points)
                    y2 = max(point[1] for point in polygon_points)
                    
                    # Extract confidence
                    confidence = text_conf_tuple[1] if isinstance(text_conf_tuple, (tuple, list)) and len(text_conf_tuple) > 1 else 1.0
                    
                    # Ensure box dimensions are valid
                    if x1 < x2 and y1 < y2:
                        # Clamp to image bounds
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        # Create box in our format: [confidence, x1, y1, x2, y2]
                        processed_boxes.append([float(confidence), float(x1), float(y1), float(x2), float(y2)])
                except Exception as e:
                    print(f"Error processing box: {e}")
                    continue
        
        return processed_boxes

    def _load_image_from_sample(self, sample):
        """Load image from sample data"""
        try:
            # Try to load from 'image' if it exists as an image
            if 'image' in sample and hasattr(sample['image'], 'convert'):
                # Image is already loaded
                pil_image = sample['image'].convert('RGB')
                return np.array(pil_image)
            
            # Try to load from 'image_path' if it exists
            elif 'image_path' in sample:
                image_path = sample['image_path']
                # If it's a URL, download it
                if image_path.startswith(('http://', 'https://')):
                    response = requests.get(image_path, stream=True)
                    response.raise_for_status()
                    pil_image = Image.open(io.BytesIO(response.content)).convert('RGB')
                    return np.array(pil_image)
                # If it's a local path
                else:
                    pil_image = Image.open(image_path).convert('RGB')
                    return np.array(pil_image)
            
            # Try to load from 'image_url' if it exists
            elif 'image_url' in sample:
                image_url = sample['image_url']
                response = requests.get(image_url, stream=True)
                response.raise_for_status()
                pil_image = Image.open(io.BytesIO(response.content)).convert('RGB')
                return np.array(pil_image)
                
            # For the invoices-and-receipts_ocr_v2 dataset specifically
            elif 'image' in sample and isinstance(sample['image'], dict) and 'bytes' in sample['image']:
                # Load image from bytes
                image_bytes = sample['image']['bytes']
                pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
                return np.array(pil_image)
                
            else:
                raise ValueError(f"Cannot find image data in sample: {sample.keys()}")
                
        except Exception as e:
            print(f"Error loading image: {e}")
            # Return a small black image as fallback
            return np.zeros((256, 256, 3), dtype=np.uint8)

    def _extract_bboxes(self, sample, image_shape):
        """Extract bounding boxes from sample annotations"""
        h, w = image_shape
        bboxes = []
        labels = []
        confidences = []
        
        # For the invoices-and-receipts_ocr_v2 dataset
        if 'annotations' in sample and isinstance(sample['annotations'], list):
            for ann in sample['annotations']:
                if isinstance(ann, dict) and 'bbox' in ann:
                    # Expected format: [x_min, y_min, width, height]
                    bbox = ann['bbox']
                    if len(bbox) == 4:
                        x_min, y_min, width, height = bbox
                        x_max = x_min + width
                        y_max = y_min + height
                        
                        # Ensure coordinates are within image bounds
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(w, x_max)
                        y_max = min(h, y_max)
                        
                        # Skip invalid boxes
                        if x_min >= x_max or y_min >= y_max:
                            continue
                        
                        bboxes.append([x_min, y_min, x_max, y_max])
                        labels.append(1)  # 1 for text
                        confidences.append(ann.get('confidence', 1.0))
        
        # Alternative format: ocr_boxes and ocr_words
        elif 'ocr_boxes' in sample and isinstance(sample['ocr_boxes'], list):
            for box in sample['ocr_boxes']:
                if len(box) >= 4:
                    x_min, y_min, x_max, y_max = box[:4]
                    confidence = box[4] if len(box) > 4 else 1.0
                    
                    # Ensure coordinates are within image bounds
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(w, x_max)
                    y_max = min(h, y_max)
                    
                    # Skip invalid boxes
                    if x_min >= x_max or y_min >= y_max:
                        continue
                    
                    bboxes.append([x_min, y_min, x_max, y_max])
                    labels.append(1)  # 1 for text
                    confidences.append(confidence)
        
        return bboxes, labels, confidences


def get_data_loaders(dataset_name="mychen76/invoices-and-receipts_ocr_v2", 
                     batch_size=8, image_size=(512, 512), num_workers=4, 
                     max_samples=None, cache_dir=None):
    """Create data loaders for training and validation"""
    
    # Simple transforms to avoid validation errors with albumentations 2.0.4
    train_transform = A.Compose([
        # Basic resize instead of RandomResizedCrop to avoid validation errors
        A.Resize(height=image_size[0], width=image_size[1], p=1.0),
        # Add some basic augmentations
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # Validation transform (just resize and normalize)
    val_transform = A.Compose([
        A.Resize(height=image_size[0], width=image_size[1], p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # Create datasets
    print("Loading training dataset...")
    train_dataset = ReceiptDataset(
        dataset_name=dataset_name, 
        split='train', 
        transform=train_transform, 
        image_size=image_size,
        max_samples=max_samples,
        cache_dir=cache_dir
    )
    
    print("Loading validation dataset...")
    val_dataset = ReceiptDataset(
        dataset_name=dataset_name, 
        split='val', 
        transform=val_transform, 
        image_size=image_size,
        max_samples=100,
        cache_dir=cache_dir
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader

def collate_fn(batch):
    """Custom collate function to handle variable number of boxes"""
    # Filter out any failed samples
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        # Return empty batch with the expected structure
        return {
            'image': torch.zeros((0, 3, 512, 512)),
            'text_map': torch.zeros((0, 1, 512, 512)),
            'boxes': [],
            'image_id': []
        }
    
    images = torch.stack([item['image'] for item in valid_batch])
    text_maps = torch.stack([item['text_map'] for item in valid_batch])
    
    # Handling variable-sized bounding boxes
    boxes = [item['boxes'] for item in valid_batch]
    image_ids = [item['image_id'] for item in valid_batch]
    
    return {
        'image': images,
        'text_map': text_maps,
        'boxes': boxes,
        'image_id': image_ids
    }


if __name__ == "__main__":
    # Test the dataset
    dataset = ReceiptDataset(max_samples=5)
    print(f"Dataset size: {len(dataset)}")
    
    # Check the first sample
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Text map shape: {sample['text_map'].shape}")
    print(f"Boxes shape: {sample['boxes'].shape}")
    
    # Test data loader
    data_loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(data_loader))
    print(f"Batch size: {batch['image'].shape[0]}")
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch text map shape: {batch['text_map'].shape}")
    print(f"Batch boxes: {[b.shape for b in batch['boxes']]}")