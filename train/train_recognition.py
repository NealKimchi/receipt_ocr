import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import json
import ast

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.data_loading import ReceiptDataset
from utils.text_utils import CharsetMapper, preprocess_text, crop_text_regions, collate_text_recognition_batch
from models.recognition.model import get_model
from models.recognition.loss import get_loss_function
from models.recognition.metrics import calculate_text_metrics, print_metrics, calculate_cer, calculate_wer

class TextRecognitionDataset(torch.utils.data.Dataset):
    """Dataset for text recognition training with real OCR text"""
    def __init__(self, 
                 dataset_name="mychen76/invoices-and-receipts_ocr_v2", 
                 split='train',
                 charset_mapper=None,
                 transform=None,
                 max_text_length=32,
                 image_height=32,
                 image_width=128,
                 keep_aspect_ratio=True,
                 max_samples=None,
                 cache_dir=None):
        """
        Initialize dataset
        Args:
            dataset_name: Name of the HuggingFace dataset
            split: 'train' or 'val'
            charset_mapper: CharsetMapper for encoding/decoding text
            transform: Image transforms
            max_text_length: Maximum text length to recognize
            image_height: Height to resize text regions to
            image_width: Width to resize text regions to
            keep_aspect_ratio: Whether to maintain aspect ratio
            max_samples: Maximum number of samples to use
            cache_dir: Cache directory for HuggingFace datasets
        """
        # Load base receipt dataset
        self.base_dataset = ReceiptDataset(
            dataset_name=dataset_name,
            split=split,
            image_size=(512, 512),  # This will be overridden for text crops
            max_samples=max_samples,
            cache_dir=cache_dir
        )
        
        # Settings
        self.charset_mapper = charset_mapper
        self.transform = transform
        self.max_text_length = max_text_length
        self.image_height = image_height
        self.image_width = image_width
        self.keep_aspect_ratio = keep_aspect_ratio
        
        # Extract text regions and annotations
        self.samples = self._extract_text_regions()
    
    def _extract_text_regions(self):
        """Extract text regions with real OCR text from base dataset"""
        samples = []
        
        print(f"Extracting text regions from {len(self.base_dataset)} images...")
        
        for idx in tqdm(range(len(self.base_dataset))):
            # Get sample from base dataset
            base_sample = self.base_dataset[idx]
            image = base_sample['image']
            boxes = base_sample['boxes']
            
            # Try to extract OCR data if available
            ocr_words = None
            ocr_boxes = None
            try:
                if 'raw_data' in base_sample and base_sample['raw_data']:
                    # Parse the raw_data JSON if it's a string
                    if isinstance(base_sample['raw_data'], str):
                        raw_data = json.loads(base_sample['raw_data'])
                    else:
                        raw_data = base_sample['raw_data']
                    
                    # Extract OCR words and boxes
                    if 'ocr_words' in raw_data:
                        # Handle different formats
                        if isinstance(raw_data['ocr_words'], str):
                            # Handle string representation of list
                            if raw_data['ocr_words'].startswith('[') and raw_data['ocr_words'].endswith(']'):
                                ocr_words = ast.literal_eval(raw_data['ocr_words'])
                            else:
                                ocr_words = [raw_data['ocr_words']]
                        else:
                            ocr_words = raw_data['ocr_words']
                    
                    if 'ocr_boxes' in raw_data:
                        # Handle different formats
                        if isinstance(raw_data['ocr_boxes'], str):
                            if raw_data['ocr_boxes'].startswith('[') and raw_data['ocr_boxes'].endswith(']'):
                                ocr_boxes = ast.literal_eval(raw_data['ocr_boxes'])
                            else:
                                # Try to handle other formats
                                try:
                                    ocr_boxes = json.loads(raw_data['ocr_boxes'])
                                except:
                                    # If all else fails, use as is
                                    ocr_boxes = raw_data['ocr_boxes']
                        else:
                            ocr_boxes = raw_data['ocr_boxes']
            except Exception as e:
                print(f"Warning: Error parsing OCR data for sample {idx}: {e}")
                ocr_words = None
                ocr_boxes = None
            
            # Convert image from tensor to numpy for cropping
            if isinstance(image, torch.Tensor):
                # Denormalize
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image.permute(1, 2, 0).numpy()
                image_np = std * image_np + mean
                image_np = np.clip(image_np, 0, 1) * 255
                image_np = image_np.astype(np.uint8)
            else:
                image_np = image
            
            # Get original image size
            h, w = image_np.shape[:2]
            
            # Crop text regions
            for i, box in enumerate(boxes):
                # Convert normalized box to pixel coordinates
                x1, y1, x2, y2 = box
                x1_px, y1_px = int(x1 * w), int(y1 * h)
                x2_px, y2_px = int(x2 * w), int(y2 * h)
                
                # Ensure valid box
                if x2_px <= x1_px or y2_px <= y1_px:
                    continue
                
                # Add some padding
                padding = 2
                x1_px = max(0, x1_px - padding)
                y1_px = max(0, y1_px - padding)
                x2_px = min(w, x2_px + padding)
                y2_px = min(h, y2_px + padding)
                
                # Crop region
                crop = image_np[y1_px:y2_px, x1_px:x2_px]
                
                # Skip if crop is too small
                if crop.shape[0] < 5 or crop.shape[1] < 5:
                    continue
                
                # Try to find matching OCR text for this crop
                ocr_text = None
                
                if ocr_words and ocr_boxes and i < len(ocr_words):
                    # Use OCR text directly if available and indices match
                    ocr_text = ocr_words[i]
                else:
                    # Fallback to generic text
                    if i < len(boxes):
                        # Use a meaningful default that's different for each box
                        ocr_text = f"item_{idx}_{i}"
                    else:
                        ocr_text = f"text_{idx}_{i}"
                
                # Add to samples
                samples.append({
                    'image': crop,
                    'text': ocr_text,
                    'box': (x1_px, y1_px, x2_px, y2_px),
                    'image_id': f"{base_sample['image_id']}_{i}"
                })
        
        print(f"Extracted {len(samples)} text regions")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample['image']
        text = sample['text']
        
        # Resize image
        if self.keep_aspect_ratio:
            # Keep aspect ratio but ensure height is fixed
            h, w = image.shape[:2]
            new_h = self.image_height
            new_w = int(w * (new_h / h))
            
            # Limit max width
            new_w = min(new_w, self.image_width)
            
            # Resize
            image = cv2.resize(image, (new_w, new_h))
            
            # Create target-sized image (pad with zeros)
            padded_image = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            padded_image[:, :new_w] = image
            image = padded_image
        else:
            # Just resize to target size
            image = cv2.resize(image, (self.image_width, self.image_height))
        
        # Normalize and convert to tensor
        if self.transform:
            image = self.transform(image=image)['image']
        else:
            # Basic normalize and convert to tensor
            image = image.astype(np.float32) / 255.0
            # Change to channels first
            image = image.transpose(2, 0, 1)
            image = torch.from_numpy(image)
        
        # Preprocess text (clean unwanted characters, normalize spacing)
        if isinstance(text, str):
            text = preprocess_text(text)
        else:
            # Handle non-string text (convert to string)
            text = preprocess_text(str(text))
        
        # Encode text if charset_mapper is provided
        if self.charset_mapper:
            encoded_text = self.charset_mapper.encode(text)
            # Important: pad with zeros if needed to reach max_text_length
            if len(encoded_text) < self.max_text_length:
                encoded_text = encoded_text + [0] * (self.max_text_length - len(encoded_text))
            # Truncate if too long    
            elif len(encoded_text) > self.max_text_length:
                encoded_text = encoded_text[:self.max_text_length]
                
            text_tensor = torch.tensor(encoded_text, dtype=torch.long)
            text_length = min(len(text), self.max_text_length)
        else:
            # Just use dummy index for proof of concept
            text_tensor = torch.zeros(self.max_text_length, dtype=torch.long)
            text_length = min(len(text), self.max_text_length)
        
        return {
            'image': image,
            'text': text_tensor,
            'raw_text': text,
            'length': text_length,
            'image_id': sample['image_id']
        }

def get_recognition_loaders(config, charset_mapper):
    """Create dataloaders for text recognition"""
    from albumentations import Compose, Normalize, Resize
    from albumentations.pytorch import ToTensorV2
    
    # Define transforms
    train_transform = Compose([
        Resize(height=config['data']['image_height'], width=config['data']['image_width']),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    val_transform = Compose([
        Resize(height=config['data']['image_height'], width=config['data']['image_width']),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    # Create datasets
    train_dataset = TextRecognitionDataset(
        dataset_name=config['data']['dataset_name'],
        split='train',
        charset_mapper=charset_mapper,
        transform=train_transform,
        max_text_length=config['data']['max_text_length'],
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
        keep_aspect_ratio=config['data']['keep_aspect_ratio'],
        max_samples=config.get('max_samples', None),
        cache_dir=config['data'].get('cache_dir', None)
    )
    
    val_dataset = TextRecognitionDataset(
        dataset_name=config['data']['dataset_name'],
        split='val',
        charset_mapper=charset_mapper,
        transform=val_transform,
        max_text_length=config['data']['max_text_length'],
        image_height=config['data']['image_height'],
        image_width=config['data']['image_width'],
        keep_aspect_ratio=config['data']['keep_aspect_ratio'],
        max_samples=config.get('max_samples', None),
        cache_dir=config['data'].get('cache_dir', None)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_text_recognition_batch,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['val_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        collate_fn=collate_text_recognition_batch,
        pin_memory=True
    )
    
    return train_loader, val_loader

def visualize_predictions(images, predictions, targets, save_path=None):
    """Visualize text recognition predictions for debugging"""
    batch_size = min(len(images), 8)  # Show at most 8 examples
    
    # Create figure
    fig, axes = plt.subplots(batch_size, 1, figsize=(12, 2 * batch_size))
    
    if batch_size == 1:
        axes = [axes]
    
    # Plot each image with prediction and target
    for i in range(batch_size):
        # Get image
        img = images[i]
        
        # Convert to numpy for visualization
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
        
        # Display image
        axes[i].imshow(img)
        
        # Set title with prediction and ground truth
        axes[i].set_title(f"Pred: '{predictions[i]}' | GT: '{targets[i]}'")
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, charset_mapper):
    """Train for one epoch with improved error handling and metrics"""
    model.train()
    epoch_loss = 0
    
    start_time = time.time()
    
    # Initialize metrics
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    batch_losses = []
    
    # Create progress bar
    num_batches = len(train_loader)
    pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Training Epoch {epoch}")
    
    for batch_idx, batch in pbar:
        try:
            # Move data to device
            images = batch['image'].to(device)
            
            # Make sure text is a tensor before moving to device
            if isinstance(batch['text'], list):
                # Convert list to tensor
                texts = torch.tensor(batch['text'], dtype=torch.long).to(device)
            else:
                texts = batch['text'].to(device)
            
            # Make sure length is a tensor
            if isinstance(batch['length'], list):
                lengths = torch.tensor(batch['length'], dtype=torch.long).to(device)
            else:
                lengths = batch['length'].to(device)
            
            # Get raw texts for debugging
            raw_texts = batch['raw_text']
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, texts, lengths)
            
            # Unpack loss if it's a dictionary
            if isinstance(loss, dict):
                total_loss = loss['loss']
            else:
                total_loss = loss
            
            # Skip potentially problematic batches
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN or Inf loss detected in batch {batch_idx}. Skipping.")
                continue
                
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping if configured
            if config['training'].get('gradient_clipping', False):
                clip_value = config['training'].get('clip_value', 5.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            # Update weights
            optimizer.step()
            
            # Record loss
            batch_loss = total_loss.item()
            epoch_loss += batch_loss
            batch_losses.append(batch_loss)
            
            # Calculate metrics
            predictions = outputs['predictions']
            for i in range(len(texts)):
                if i >= len(raw_texts):
                    continue
                    
                # Get prediction (using CTC decoding - remove repeated characters and blanks)
                pred_indices = predictions[i].cpu().numpy()
                # Get unique consecutive characters (CTC decoding)
                filtered_indices = []
                for j, idx in enumerate(pred_indices):
                    if idx != 0 and (j == 0 or idx != pred_indices[j-1]):
                        filtered_indices.append(idx)
                
                pred_text = charset_mapper.decode(filtered_indices)
                
                # Get target text
                if raw_texts[i]:
                    target_text = raw_texts[i]
                else:
                    # Use encoded text if raw text is not available
                    if texts.dim() == 2 and i < texts.size(0):
                        target_indices = texts[i, :lengths[i]].cpu().numpy()
                        target_text = charset_mapper.decode(target_indices)
                    else:
                        target_text = ""
                
                # Store for metrics calculation
                all_predictions.append(pred_text)
                all_targets.append(target_text)
                
                # Incremental accuracy with different strictness levels
                if pred_text.lower() == target_text.lower():
                    # Exact match ignoring case
                    correct += 1
                elif len(pred_text) > 0 and len(target_text) > 0:
                    # Calculate similarity based on edit distance
                    distance = calculate_cer(pred_text, target_text)
                    if distance < 0.3:  # More than 70% similar
                        correct += 0.5  # Partial credit
                
                total += 1
            
            # Update progress bar with latest metrics
            accuracy = correct / total if total > 0 else 0
            pbar.set_postfix({
                'loss': f"{batch_loss:.4f}",
                'acc': f"{accuracy:.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Print detailed metrics at intervals
            log_interval = config['training']['log_interval']
            if (batch_idx + 1) % log_interval == 0 or batch_idx == num_batches - 1:
                # Print some example predictions
                print("\nSample predictions:")
                for i in range(min(3, len(images))):
                    if i < len(all_predictions) and i < len(all_targets):
                        print(f"  Pred: '{all_predictions[-i-1]}'")
                        print(f"  True: '{all_targets[-i-1]}'")
                        print()
            
            # Visualize predictions periodically
            vis_interval = config['training']['visualization_interval']
            if (batch_idx + 1) % vis_interval == 0:
                try:
                    # Get most recent predictions for visualization
                    max_vis = min(8, len(predictions))
                    pred_texts = all_predictions[-max_vis:]
                    target_texts = all_targets[-max_vis:]
                    vis_images = images[:max_vis]
                    
                    # Save visualization
                    vis_path = os.path.join(config['paths']['output_dir'], f"train_vis_epoch{epoch}_batch{batch_idx}.png")
                    visualize_predictions(vis_images, pred_texts, target_texts, save_path=vis_path)
                except Exception as e:
                    print(f"Warning: Visualization failed - {e}")
        
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory in batch {batch_idx}. Skipping batch and freeing memory.")
                # Free memory and continue
                torch.cuda.empty_cache()
                continue
            else:
                print(f"Runtime error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
                
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Calculate final metrics
    case_sensitive = config['evaluation'].get('case_sensitive', False)
    ignore_punctuation = config['evaluation'].get('ignore_punctuation', True)
    
    # Skip metrics calculation if we have no predictions
    if len(all_predictions) == 0 or len(all_targets) == 0:
        print("Warning: No valid predictions or targets for metrics calculation.")
        metrics = {'cer': 1.0, 'wer': 1.0, 'accuracy': 0.0, 'num_samples': 0}
    else:
        metrics = calculate_text_metrics(all_predictions, all_targets, case_sensitive, ignore_punctuation)
    
    # Average training loss
    avg_train_loss = epoch_loss / (len(train_loader) - len(train_loader) // 10) if len(train_loader) > 0 else float('inf')
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    
    # Print epoch summary
    elapsed_time = time.time() - start_time
    print(f"\nEpoch {epoch} completed in {elapsed_time:.2f}s")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    print(f"CER: {metrics['cer']:.4f}, WER: {metrics['wer']:.4f}")
    
    return avg_train_loss, accuracy, metrics

def validate(model, val_loader, criterion, device, epoch, config, charset_mapper):
    """Validate the model"""
    model.eval()
    val_loss = 0
    all_predictions = []
    all_targets = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}")):
            # Move data to device
            images = batch['image'].to(device)
            
            # Make sure text is a tensor before moving to device
            if isinstance(batch['text'], list):
                # Convert list to tensor
                texts = torch.tensor(batch['text'], dtype=torch.long).to(device)
            else:
                texts = batch['text'].to(device)
            
            # Make sure length is a tensor
            if isinstance(batch['length'], list):
                lengths = torch.tensor(batch['length'], dtype=torch.long).to(device)
            else:
                lengths = batch['length'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, texts, lengths)
            
            # Unpack loss if it's a dictionary
            if isinstance(loss, dict):
                total_loss = loss['loss']
            else:
                total_loss = loss
            
            val_loss += total_loss.item()
            
            # Get predictions and targets for metrics
            predictions = outputs['predictions']
            for i in range(len(texts)):
                # Get prediction
                pred_text = charset_mapper.decode(predictions[i])
                
                # Handle different tensor formats for texts
                if texts.dim() == 2:
                    # If texts is a 2D tensor (batch_size, seq_length)
                    target_text = charset_mapper.decode(texts[i][:lengths[i]])
                else:
                    # If somehow texts is a 1D tensor or other format
                    target_text = charset_mapper.decode(texts[i])
                
                all_predictions.append(pred_text)
                all_targets.append(target_text)
            
            # Visualize predictions on first validation batch
            if batch_idx == 0:
                try:
                    # Get some predictions to visualize
                    pred_texts = [charset_mapper.decode(pred) for pred in predictions[:8]]
                    
                    if texts.dim() == 2:
                        target_texts = [charset_mapper.decode(texts[i][:lengths[i]]) for i in range(min(8, len(texts)))]
                    else:
                        target_texts = [charset_mapper.decode(texts[i]) for i in range(min(8, len(texts)))]
                    
                    # Save visualization
                    vis_path = os.path.join(config['paths']['output_dir'], f"val_vis_epoch{epoch}.png")
                    visualize_predictions(images[:8], pred_texts, target_texts, save_path=vis_path)
                except Exception as e:
                    print(f"Warning: Validation visualization failed - {e}")
    
    # Average validation loss
    avg_val_loss = val_loss / len(val_loader)
    
    # Calculate metrics
    case_sensitive = config['evaluation'].get('case_sensitive', False)
    ignore_punctuation = config['evaluation'].get('ignore_punctuation', True)
    metrics = calculate_text_metrics(all_predictions, all_targets, case_sensitive, ignore_punctuation)
    
    # Print validation summary
    elapsed_time = time.time() - start_time
    print(f"Validation completed in {elapsed_time:.2f}s - "
          f"Val Loss: {avg_val_loss:.4f}")
    print_metrics(metrics)
    
    return avg_val_loss, metrics

def train_model(config, output_dir):
    """Train the text recognition model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create character set mapper
    charset_mapper = CharsetMapper(config['data']['alphabet_path'])
    print(f"Loaded charset with {charset_mapper.charset_size} characters")
    
    # Get data loaders
    train_loader, val_loader = get_recognition_loaders(config, charset_mapper)
    
    # Create model
    model = get_model(config, num_classes=charset_mapper.charset_size + 1)  # +1 for blank/CTC
    model = model.to(device)
    
    # Print model summary
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Create loss function
    criterion = get_loss_function(config)
    
    # Create optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=(config['training'].get('beta1', 0.9), config['training'].get('beta2', 0.999))
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training'].get('momentum', 0.9),
            weight_decay=config['training'].get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']}")
    
    # Create learning rate scheduler
    if config['training'].get('use_scheduler', False):
        scheduler_type = config['training'].get('lr_scheduler', 'step')
        
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['training']['lr_step_size'],
                gamma=config['training']['lr_gamma']
            )
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config['training']['epochs'],
                eta_min=config['training'].get('min_lr', 1e-6)
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=config['training'].get('patience', 5),
                min_lr=config['training'].get('min_lr', 1e-6)
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    # Track best model
    best_val_loss = float('inf')
    best_accuracy = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    best_acc_model_path = os.path.join(output_dir, 'best_accuracy_model.pth')
    
    # Initialize metrics tracking
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_metrics_history = []
    
    # Train for specified number of epochs
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train for one epoch
        # FIX: Changed to unpack three values - train_loss, train_accuracy, and train_metrics
        train_loss, train_accuracy, train_metrics = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            charset_mapper=charset_mapper
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            config=config,
            charset_mapper=charset_mapper
        )
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
            print(f"Learning rate: {current_lr:.6f}")
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_accuracy)
        val_metrics_history.append(val_metrics)
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Save best model based on sequence accuracy
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            torch.save(model.state_dict(), best_acc_model_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")
        
        # Plot training curve
        if epoch % 5 == 0 or epoch == config['training']['epochs']:
            plot_training_curve(
                train_losses=train_losses,
                val_losses=val_losses,
                train_accuracies=train_accuracies,
                val_metrics=val_metrics_history,
                output_dir=output_dir,
                epoch=epoch
            )
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch{epoch}.pth')
        if epoch % config['training']['save_interval'] == 0 or epoch == config['training']['epochs']:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        
        # Early stopping
        if config['training'].get('early_stopping', False):
            patience = config['training'].get('patience', 10)
            if len(val_losses) > patience:
                # Check if validation loss hasn't improved for `patience` epochs
                if all(val_losses[-i-1] <= val_losses[-i] for i in range(patience)):
                    print(f"Early stopping at epoch {epoch} as validation loss hasn't improved for {patience} epochs")
                    break
    
    # Return training results
    return {
        'best_val_loss': best_val_loss,
        'best_accuracy': best_accuracy,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics_history
    }

def plot_training_curve(train_losses, val_losses, train_accuracies, val_metrics, output_dir, epoch):
    """Plot training curves"""
    # Create figure with 2 rows of plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot train accuracy
    ax2.plot(epochs, train_accuracies, 'g-')
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)
    
    # Plot validation metrics
    val_accuracies = [m['accuracy'] for m in val_metrics]
    val_cer = [m['cer'] for m in val_metrics]
    val_wer = [m['wer'] for m in val_metrics]
    
    ax3.plot(epochs, val_accuracies, 'm-')
    ax3.set_title('Validation Sequence Accuracy')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True)
    
    ax4.plot(epochs, val_cer, 'c-', label='CER')
    ax4.plot(epochs, val_wer, 'y-', label='WER')
    ax4.set_title('Validation Error Rates')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Error Rate')
    ax4.legend()
    ax4.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curve_epoch{epoch}.png'))
    plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Receipt Text Recognition Model')
    parser.add_argument('--config', type=str, default='configs/recognition_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs/recognition',
                        help='Directory to save outputs')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Update config with output directory
    config['paths']['output_dir'] = output_dir
    
    # Save config to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Train model
    train_results = train_model(config, output_dir)
    
    # Save results summary
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"Best validation loss: {train_results['best_val_loss']:.4f}\n")
        f.write(f"Best accuracy: {train_results['best_accuracy']:.4f}\n")
        f.write("Final metrics:\n")
        for k, v in train_results['val_metrics_history'][-1].items():
            if isinstance(v, float):
                f.write(f"  {k}: {v:.4f}\n")
            else:
                f.write(f"  {k}: {v}\n")

if __name__ == "__main__":
    main()