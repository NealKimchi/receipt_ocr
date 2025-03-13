#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training script for text detection model using Hugging Face dataset.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
from models.detection.model import get_model
from models.detection.loss import get_loss_function
from utils.data_loading import get_data_loaders


def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, 
                device, num_epochs=50, save_dir='checkpoints', experiment_name='text_detection'):
    """
    Train the text detection model
    
    Args:
        model: PyTorch model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        num_epochs: Number of epochs to train
        save_dir: Directory to save model checkpoints
        experiment_name: Name of the experiment for logging
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(save_dir, f"{experiment_name}_log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Starting training for {experiment_name}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write("=" * 50 + "\n")
    
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'text_map_precision': [],
        'text_map_recall': [],
        'text_map_f1': [],
        'box_precision': [],
        'box_recall': [],
        'box_f1': []
    }
    
    # Track best validation loss
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_text_map_loss = 0.0
        train_box_loss = 0.0
        train_confidence_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for batch in progress_bar:
            # Move data to device
            images = batch['image'].to(device)
            text_maps = batch['text_map'].to(device)
            
            # Debug: Print image shape
            print(f"**DEBUG**\tInput image shape: {images.shape}")
            
            # Forward pass
            predictions = model(images)
            
            # Calculate loss
            loss_dict = loss_fn(predictions, {
                'text_map': text_maps,
                'boxes': batch['boxes']
            })
            
            loss = loss_dict['total_loss']
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            train_text_map_loss += loss_dict['text_map_loss'].item()
            train_box_loss += loss_dict['box_loss'].item()
            train_confidence_loss += loss_dict['confidence_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'text_loss': loss_dict['text_map_loss'].item(),
                'box_loss': loss_dict['box_loss'].item()
            })
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        train_text_map_loss /= len(train_loader)
        train_box_loss /= len(train_loader)
        train_confidence_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_text_map_loss = 0.0
        val_box_loss = 0.0
        val_confidence_loss = 0.0
        
        # Metrics for segmentation
        all_preds = []
        all_targets = []
        
        # Metrics for box detection
        all_pred_boxes = []
        all_target_boxes = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for batch in progress_bar:
                # Move data to device
                images = batch['image'].to(device)
                text_maps = batch['text_map'].to(device)
                
                # Forward pass
                predictions = model(images)
                
                # Calculate loss
                loss_dict = loss_fn(predictions, {
                    'text_map': text_maps,
                    'boxes': batch['boxes']
                })
                
                loss = loss_dict['total_loss']
                
                # Update metrics
                val_loss += loss.item()
                val_text_map_loss += loss_dict['text_map_loss'].item()
                val_box_loss += loss_dict['box_loss'].item()
                val_confidence_loss += loss_dict['confidence_loss'].item()
                
                # Collect predictions and targets for metrics
                pred_maps = (predictions['text_map'] > 0.5).float()
                all_preds.append(pred_maps.cpu())
                all_targets.append(text_maps.cpu())
                
                # Collect box predictions
                for i in range(len(images)):
                    # Get prediction map
                    pred_conf = predictions['confidence'][i]  # (1, H, W)
                    pred_boxes = predictions['bbox_coords'][i]  # (4, H, W)
                    
                    # Extract boxes using non-maximum suppression
                    detected_boxes = extract_boxes(pred_conf, pred_boxes, threshold=0.5)
                    all_pred_boxes.append(detected_boxes)
                    
                    # Target boxes
                    target_boxes = batch['boxes'][i]
                    all_target_boxes.append(target_boxes)
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        val_text_map_loss /= len(val_loader)
        val_box_loss /= len(val_loader)
        val_confidence_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Calculate text map metrics
        text_precision, text_recall, text_f1 = calculate_segmentation_metrics(all_preds, all_targets)
        
        # Calculate box metrics
        box_precision, box_recall, box_f1 = calculate_box_metrics(all_pred_boxes, all_target_boxes)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['text_map_precision'].append(text_precision)
        history['text_map_recall'].append(text_recall)
        history['text_map_f1'].append(text_f1)
        history['box_precision'].append(box_precision)
        history['box_recall'].append(box_recall)
        history['box_f1'].append(box_f1)
        
        # Print metrics
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Text Map - Precision: {text_precision:.4f}, Recall: {text_recall:.4f}, F1: {text_f1:.4f}")
        print(f"Boxes - Precision: {box_precision:.4f}, Recall: {box_recall:.4f}, F1: {box_f1:.4f}")
        
        # Log metrics
        with open(log_file, 'a') as f:
            f.write(f"Epoch {epoch+1}/{num_epochs}\n")
            f.write(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
            f.write(f"Text Map - Precision: {text_precision:.4f}, Recall: {text_recall:.4f}, F1: {text_f1:.4f}\n")
            f.write(f"Boxes - Precision: {box_precision:.4f}, Recall: {box_recall:.4f}, F1: {box_f1:.4f}\n")
            f.write("-" * 50 + "\n")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, os.path.join(save_dir, f"{experiment_name}_best.pth"))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, os.path.join(save_dir, f"{experiment_name}_epoch{epoch+1}.pth"))
        
        # Visualize some predictions
        if (epoch + 1) % 5 == 0:
            visualize_predictions(model, val_loader, device, 
                                  os.path.join(save_dir, f"viz_epoch{epoch+1}"), 
                                  num_samples=5)
    
    # Plot training history
    plot_training_history(history, os.path.join(save_dir, f"{experiment_name}_history.png"))
    
    return model, history


def extract_boxes(confidence_map, box_coords, threshold=0.5, nms_threshold=0.3):
    """
    Extract bounding boxes from the model output using confidence threshold and NMS
    
    Args:
        confidence_map (torch.Tensor): Predicted confidence map (1, H, W)
        box_coords (torch.Tensor): Predicted box coordinates (4, H, W)
        threshold (float): Confidence threshold
        nms_threshold (float): IoU threshold for NMS
    
    Returns:
        list: List of detected boxes [x1, y1, x2, y2, confidence]
    """
    # Get dimensions
    h, w = confidence_map.shape[1:]
    
    # Threshold confidence map
    binary_map = (confidence_map > threshold).float()[0]  # (H, W)
    
    # If no detections, return empty list
    if binary_map.sum() == 0:
        return []
    
    # Get indices of high confidence points
    y_indices, x_indices = torch.where(binary_map > 0)
    
    # Get box coordinates and confidence at those points
    boxes = []
    for i in range(len(y_indices)):
        y, x = y_indices[i], x_indices[i]
        
        # Get predicted box coordinates
        box = box_coords[:, y, x].cpu().numpy()  # (4,)
        
        # Convert to absolute coordinates
        x1, y1, x2, y2 = box
        x1 = max(0, x1 * w)
        y1 = max(0, y1 * h)
        x2 = min(w, x2 * w)
        y2 = min(h, y2 * h)
        
        # Get confidence score
        conf = confidence_map[0, y, x].item()
        
        boxes.append([x1, y1, x2, y2, conf])
    
    # Perform non-maximum suppression
    boxes = np.array(boxes)
    if len(boxes) > 0:
        # Sort by confidence
        indices = np.argsort(-boxes[:, 4])
        boxes = boxes[indices]
        
        # NMS
        keep = []
        while len(boxes) > 0:
            keep.append(boxes[0])
            if len(boxes) == 1:
                break
            
            # Calculate IoU of the first box with all other boxes
            ious = calculate_iou(boxes[0, :4], boxes[1:, :4])
            
            # Find boxes with IoU less than threshold
            mask = ious < nms_threshold
            boxes = boxes[1:][mask]
        
        boxes = np.array(keep)
    
    return boxes


def calculate_iou(box, boxes):
    """
    Calculate IoU between a box and multiple boxes
    
    Args:
        box (numpy.ndarray): Single box [x1, y1, x2, y2]
        boxes (numpy.ndarray): Multiple boxes (N, 4)
    
    Returns:
        numpy.ndarray: IoU values (N,)
    """
    # Calculate intersection
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    # Intersection area
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Box areas
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # IoU
    iou = intersection / (box_area + boxes_area - intersection + 1e-6)
    
    return iou


def calculate_segmentation_metrics(predictions, targets):
    """
    Calculate precision, recall, and F1 score for text segmentation
    
    Args:
        predictions (list): List of predicted binary masks
        targets (list): List of target binary masks
    
    Returns:
        tuple: (precision, recall, f1)
    """
    # Concatenate all predictions and targets
    preds = torch.cat([p.flatten() for p in predictions])
    targs = torch.cat([t.flatten() for t in targets])
    
    # Convert to binary
    preds = (preds > 0.5).float()
    targs = (targs > 0.5).float()
    
    # Calculate metrics
    true_positives = (preds * targs).sum().item()
    false_positives = (preds * (1 - targs)).sum().item()
    false_negatives = ((1 - preds) * targs).sum().item()
    
    # Calculate precision, recall, and F1
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision, recall, f1


def calculate_box_metrics(pred_boxes, target_boxes, iou_threshold=0.5):
    """
    Calculate precision, recall, and F1 score for box detection
    
    Args:
        pred_boxes (list): List of predicted boxes
        target_boxes (list): List of target boxes
        iou_threshold (float): IoU threshold for a true positive
    
    Returns:
        tuple: (precision, recall, f1)
    """
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    for i in range(len(pred_boxes)):
        preds = pred_boxes[i]
        targets = target_boxes[i]
        
        if len(targets) == 0:
            # If no targets, all predictions are false positives
            total_false_positives += len(preds)
            continue
        
        if len(preds) == 0:
            # If no predictions, all targets are false negatives
            total_false_negatives += len(targets)
            continue
        
        # Convert targets to expected format
        if isinstance(targets, torch.Tensor):
            # Handle tensor format
            targets = targets.cpu().numpy()
            if targets.shape[1] > 4:  # If targets include confidence
                targets = targets[:, 1:5]  # Keep only box coordinates
        
        # Calculate IoU between each prediction and target
        ious = np.zeros((len(preds), len(targets)))
        for j, pred in enumerate(preds):
            for k, target in enumerate(targets):
                # Convert prediction format if needed
                if len(pred) > 4:  # If prediction includes confidence
                    pred_box = pred[:4]
                else:
                    pred_box = pred
                
                # Calculate IoU
                iou = calculate_single_iou(pred_box, target)
                ious[j, k] = iou
        
        # Match predictions to targets
        matched_targets = set()
        for j in range(len(preds)):
            # If no valid match, it's a false positive
            if ious[j].max() < iou_threshold:
                total_false_positives += 1
                continue
            
            # Find the best matching target
            best_target = ious[j].argmax()
            
            # If the target is already matched, it's a false positive
            if best_target in matched_targets:
                total_false_positives += 1
                continue
            
            # It's a true positive
            matched_targets.add(best_target)
            total_true_positives += 1
        
        # Unmatched targets are false negatives
        total_false_negatives += len(targets) - len(matched_targets)
    
    # Calculate metrics
    precision = total_true_positives / (total_true_positives + total_false_positives + 1e-6)
    recall = total_true_positives / (total_true_positives + total_false_negatives + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    
    return precision, recall, f1


def calculate_single_iou(box1, box2):
    """Calculate IoU between two boxes"""
    # Box1
    x1, y1, x2, y2 = box1
    box1_area = (x2 - x1) * (y2 - y1)
    
    # Box2
    x1_t, y1_t, x2_t, y2_t = box2
    box2_area = (x2_t - x1_t) * (y2_t - y1_t)
    
    # Intersection
    x1_i = max(x1, x1_t)
    y1_i = max(y1, y1_t)
    x2_i = min(x2, x2_t)
    y2_i = min(y2, y2_t)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    union = box1_area + box2_area - intersection
    
    # IoU
    iou = intersection / (union + 1e-6)
    
    return iou


def visualize_predictions(model, data_loader, device, save_dir, num_samples=5):
    """Visualize model predictions on some samples"""
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break
            
            # Move data to device
            images = batch['image'].to(device)
            
            # Make predictions
            predictions = model(images)
            
            # Process each sample in the batch
            for j in range(len(images)):
                # Get image and predictions
                image = images[j].cpu().permute(1, 2, 0).numpy()
                
                # Denormalize image
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image = (image * std + mean) * 255
                image = image.astype(np.uint8)
                
                # Get text map prediction
                text_map = predictions['text_map'][j][0].cpu().numpy()
                
                # Get confidence and box predictions
                conf_map = predictions['confidence'][j][0].cpu().numpy()
                box_coords = predictions['bbox_coords'][j].cpu().numpy()
                
                # Extract boxes
                detected_boxes = extract_boxes(
                    predictions['confidence'][j].unsqueeze(0),
                    predictions['bbox_coords'][j],
                    threshold=0.5
                )
                
                # Create visualization
                # 1. Original image with detected boxes
                img_with_boxes = Image.fromarray(image)
                draw = ImageDraw.Draw(img_with_boxes)
                
                # Draw target boxes in green
                target_boxes = batch['boxes'][j]
                if isinstance(target_boxes, torch.Tensor) and len(target_boxes) > 0:
                    for box in target_boxes:
                        # Target boxes format: [conf, x1, y1, x2, y2]
                        x1, y1, x2, y2 = box[1:5].numpy()
                        draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
                
                # Draw predicted boxes in red
                for box in detected_boxes:
                    x1, y1, x2, y2, conf = box
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
                    draw.text((x1, y1 - 10), f"{conf:.2f}", fill='red')
                
                # Save visualizations
                img_with_boxes.save(os.path.join(save_dir, f"sample_{i}_{j}_boxes.png"))
                
                # 2. Text map visualization
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(text_map, cmap='jet', alpha=0.7)
                plt.title("Text Map Prediction")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{i}_{j}_text_map.png"))
                plt.close()
                
                # 3. Confidence map visualization
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title("Original Image")
                plt.axis('off')
                
                plt.subplot(1, 2, 2)
                plt.imshow(conf_map, cmap='jet', alpha=0.7)
                plt.title("Confidence Map")
                plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"sample_{i}_{j}_conf_map.png"))
                plt.close()


def plot_training_history(history, save_path):
    """Plot training history"""
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Text map metrics
    plt.subplot(2, 2, 2)
    plt.plot(history['text_map_precision'], label='Precision')
    plt.plot(history['text_map_recall'], label='Recall')
    plt.plot(history['text_map_f1'], label='F1 Score')
    plt.title('Text Map Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    # Box metrics
    plt.subplot(2, 2, 3)
    plt.plot(history['box_precision'], label='Precision')
    plt.plot(history['box_recall'], label='Recall')
    plt.plot(history['box_f1'], label='F1 Score')
    plt.title('Box Detection Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    """Main training function with hardcoded parameters"""
    # Hardcoded configuration
    hf_dataset = "mychen76/invoices-and-receipts_ocr_v2"
    save_dir = "detection_model_output"
    experiment_name = "text_detection"
    batch_size = 4
    num_epochs = 50
    learning_rate = 0.001
    image_size = (512, 512)
    max_samples = None  # Set to a number for debugging (e.g., 100)
    
    print(f"Using Hugging Face dataset: {hf_dataset}")
    print(f"Saving results to: {save_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        dataset_name=hf_dataset,
        batch_size=batch_size,
        image_size=image_size,
        max_samples=max_samples
    )
    
    # Create model
    print("Creating model...")
    model = get_model(in_channels=3, out_channels=1).to(device)
    
    # Create loss function
    loss_fn = get_loss_function(text_map_weight=1.0, box_weight=1.0, confidence_weight=0.5)
    
    # Create optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    
    # Create scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Train model
    print("Starting training...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        save_dir=save_dir,
        experiment_name=experiment_name
    )
    
    print("Training completed!")
    return model, history


if __name__ == "__main__":
    main()