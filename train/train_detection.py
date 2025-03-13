import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import time
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from utils.data_loading import ReceiptDataset, get_data_loaders
from models.detection.model import get_model
from models.detection.loss import get_loss_function
from models.detection.metrics import calculate_detection_metrics

def visualize_predictions(image, outputs, batch, index, device, save_path=None):
    """Visualize predictions for debugging"""
    # Convert tensors to numpy for visualization
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    
    # Standardize image for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = std * image_np + mean
    image_np = np.clip(image_np, 0, 1)
    
    # Get text map prediction - detach from computational graph
    text_map_pred = outputs['text_map'][index, 0].detach().cpu().numpy()
    
    # Get confidence prediction - detach from computational graph
    conf_pred = outputs['confidence'][index, 0].detach().cpu().numpy()
    
    # Get box predictions - detach from computational graph
    bbox_pred = outputs['bbox_coords'][index].detach().cpu().numpy()
    
    # Get ground truth text map
    text_map_gt = batch['text_map'][index, 0].cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Predicted text map
    axes[0, 1].imshow(text_map_pred, cmap='gray')
    axes[0, 1].set_title('Predicted Text Map')
    axes[0, 1].axis('off')
    
    # Ground truth text map
    axes[0, 2].imshow(text_map_gt, cmap='gray')
    axes[0, 2].set_title('Ground Truth Text Map')
    axes[0, 2].axis('off')
    
    # Confidence prediction
    axes[1, 0].imshow(conf_pred, cmap='jet')
    axes[1, 0].set_title('Confidence Prediction')
    axes[1, 0].axis('off')
    
    # Detected boxes visualization
    h, w = text_map_pred.shape
    box_vis = image_np.copy()
    
    # Threshold text map to find regions
    binary_map = (text_map_pred > 0.3).astype(np.uint8)  # Lower threshold for better recall
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw detected regions
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Filter small contours
            x, y, w_, h_ = cv2.boundingRect(contour)
            # Convert to normalized coordinates
            x1, y1 = x / w, y / h
            x2, y2 = (x + w_) / w, (y + h_) / h
            
            # Draw rectangle on image
            cv2.rectangle(box_vis, (x, y), (x + w_, y + h_), (0, 1, 0), 2)
    
    axes[1, 1].imshow(box_vis)
    axes[1, 1].set_title('Detected Boxes')
    axes[1, 1].axis('off')
    
    # Ground truth boxes
    gt_box_vis = image_np.copy()
    
    # Get ground truth boxes
    gt_boxes = batch['boxes'][index]
    
    # Draw ground truth boxes
    for box in gt_boxes:
        if isinstance(box, torch.Tensor):
            box = box.detach().cpu().numpy()
            
        x1, y1, x2, y2 = box
        
        # Convert normalized coordinates to pixel coordinates
        x1_px, y1_px = int(x1 * w), int(y1 * h)
        x2_px, y2_px = int(x2 * w), int(y2 * h)
        
        cv2.rectangle(gt_box_vis, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 1), 2)
    
    axes[1, 2].imshow(gt_box_vis)
    axes[1, 2].set_title('Ground Truth Boxes')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def calculate_metrics(pred, target, threshold=0.3):
    """
    Calculate precision, recall, F1, and IoU for text detection
    Args:
        pred: Predicted text map (B, 1, H, W)
        target: Target text map (B, 1, H, W)
        threshold: Threshold for binarization
    Returns:
        precision, recall, f1, iou
    """
    # Ensure inputs are proper shape
    if pred.dim() > 2:
        pred = pred.view(-1)
    if target.dim() > 2:
        target = target.view(-1)
    
    # Convert to binary predictions
    pred_binary = (pred > threshold).float()
    target_binary = (target > threshold).float()
    
    # Calculate intersection and union
    intersection = (pred_binary * target_binary).sum().item()
    union = (pred_binary.sum() + target_binary.sum()).item() - intersection + 1e-8
    
    # Prevent division by zero
    if pred_binary.sum().item() == 0 and target_binary.sum().item() == 0:
        return 1.0, 1.0, 1.0, 1.0
    
    if pred_binary.sum().item() == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    if target_binary.sum().item() == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Precision, recall, F1
    precision = intersection / (pred_binary.sum().item() + 1e-8)
    recall = intersection / (target_binary.sum().item() + 1e-8)
    
    # Prevent division by zero in F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
    else:
        f1 = 0.0
        
    iou = intersection / union
    
    return precision, recall, f1, iou

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, output_dir):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    batch_losses = []
    
    # Initialize metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0
    sample_count = 0
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        # Move data to device
        images = batch['image'].to(device)
        text_maps = batch['text_map'].to(device)
        boxes = batch['boxes']  # List of lists of normalized boxes
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss_dict = criterion(outputs, {
            'text_map': text_maps,
            'boxes': boxes
        })
        
        total_loss = loss_dict['total_loss']
        
        # Backward pass and optimize
        total_loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += total_loss.item()
        batch_losses.append(total_loss.item())
        
        # Calculate metrics for this batch
        batch_precision, batch_recall, batch_f1, batch_iou = calculate_metrics(
            outputs['text_map'].detach(), text_maps, threshold=0.3
        )
        
        total_precision += batch_precision
        total_recall += batch_recall
        total_f1 += batch_f1
        total_iou += batch_iou
        sample_count += 1
        
        # Print detailed loss components periodically
        log_interval = config['training'].get('log_interval', 10)
        if (batch_idx + 1) % log_interval == 0:
            print(f"\nBatch {batch_idx+1}/{len(train_loader)}: "
                  f"Total Loss: {total_loss.item():.4f}, "
                  f"Text Map Loss: {loss_dict['text_map_loss'].item():.4f}, "
                  f"Box Loss: {loss_dict['box_loss'].item():.4f}, "
                  f"Confidence Loss: {loss_dict['confidence_loss'].item():.4f}")
            
            # Print metrics
            print(f"Batch Metrics - Precision: {batch_precision:.4f}, "
                  f"Recall: {batch_recall:.4f}, F1: {batch_f1:.4f}, IoU: {batch_iou:.4f}")
        
        # Visualize predictions periodically
        vis_interval = config['training'].get('visualization_interval', 50)
        if (batch_idx + 1) % vis_interval == 0:
            vis_path = os.path.join(output_dir, f"train_vis_epoch{epoch}_batch{batch_idx}.png")
            visualize_predictions(images[0], outputs, batch, 0, device, save_path=vis_path)
    
    # Average training loss for this epoch
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Average metrics
    avg_precision = total_precision / sample_count if sample_count > 0 else 0
    avg_recall = total_recall / sample_count if sample_count > 0 else 0
    avg_f1 = total_f1 / sample_count if sample_count > 0 else 0
    avg_iou = total_iou / sample_count if sample_count > 0 else 0
    
    # Print epoch summary
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {elapsed_time:.2f}s - "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Precision: {avg_precision:.4f}, "
          f"Recall: {avg_recall:.4f}, "
          f"F1: {avg_f1:.4f}, "
          f"IoU: {avg_iou:.4f}")
    
    # Return metrics
    metrics = {
        'loss': avg_train_loss,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'iou': avg_iou
    }
    
    return avg_train_loss, metrics, batch_losses

def validate(model, val_loader, criterion, device, epoch, config, output_dir):
    """Validate the model"""
    model.eval()
    val_loss = 0
    
    # Initialize metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0
    sample_count = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}")):
            # Move data to device
            images = batch['image'].to(device)
            text_maps = batch['text_map'].to(device)
            boxes = batch['boxes']  # List of lists of normalized boxes
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss_dict = criterion(outputs, {
                'text_map': text_maps,
                'boxes': boxes
            })
            
            val_loss += loss_dict['total_loss'].item()
            
            # Calculate metrics
            batch_precision, batch_recall, batch_f1, batch_iou = calculate_metrics(
                outputs['text_map'], text_maps, threshold=0.3  # Lower threshold for better recall
            )
            
            total_precision += batch_precision
            total_recall += batch_recall
            total_f1 += batch_f1
            total_iou += batch_iou
            sample_count += 1
            
            # Visualize predictions on first validation batch
            if batch_idx == 0:
                vis_path = os.path.join(output_dir, f"val_vis_epoch{epoch}.png")
                visualize_predictions(images[0], outputs, batch, 0, device, save_path=vis_path)
    
    # Average validation loss
    avg_val_loss = val_loss / len(val_loader)
    
    # Average metrics
    metrics = {}
    metrics['precision'] = total_precision / sample_count if sample_count > 0 else 0
    metrics['recall'] = total_recall / sample_count if sample_count > 0 else 0
    metrics['f1'] = total_f1 / sample_count if sample_count > 0 else 0
    metrics['iou'] = total_iou / sample_count if sample_count > 0 else 0
    
    # Print epoch summary
    elapsed_time = time.time() - start_time
    print(f"Validation completed in {elapsed_time:.2f}s - "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Precision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, "
          f"F1: {metrics['f1']:.4f}, "
          f"IoU: {metrics['iou']:.4f}")
    
    return avg_val_loss, metrics

def train_model(config, output_dir):
    """Train the text detection model"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, val_loader = get_data_loaders(
        dataset_name=config['data']['dataset_name'],
        batch_size=config['training']['batch_size'],
        image_size=tuple(config['data']['image_size']),
        num_workers=config['data']['num_workers'],
        max_samples=config['data'].get('max_samples', None),
        cache_dir=config['data'].get('cache_dir', None)
    )
    
    # Create model
    model = get_model(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )
    model = model.to(device)
    
    # Create loss function
    criterion = get_loss_function(
        text_map_weight=config['loss']['text_map_weight'],
        box_weight=config['loss']['box_weight'],
        confidence_weight=config['loss']['confidence_weight']
    )
    
    # Create optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-4)
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
    if config['training'].get('lr_scheduler', None) == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training'].get('min_lr', 1e-6)
        )
    elif config['training'].get('lr_scheduler', None) == 'cosine_warmup':
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['training'].get('warmup_epochs', 5),
            T_mult=config['training'].get('warmup_mult', 2),
            eta_min=config['training'].get('min_lr', 1e-6)
        )
    elif config['training'].get('lr_scheduler', None) == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('lr_step_size', 10),
            gamma=config['training'].get('lr_gamma', 0.1)
        )
    else:
        scheduler = None
    
    # Track best model
    best_val_loss = float('inf')
    best_f1 = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    # Initialize metrics tracking
    train_losses = []
    val_losses = []
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': []
    }
    
    # Train for specified number of epochs
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train for one epoch
        train_loss, train_metrics, batch_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            output_dir=output_dir
        )
        
        # Validate
        val_loss, val_metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            config=config,
            output_dir=output_dir
        )
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"Learning rate: {current_lr:.6f}")
        
        # Track metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        for metric in metrics:
            metrics[metric].append(val_metrics[metric])
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")
        
        # Save best model based on F1 score
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_f1_model.pth'))
            print(f"New best model saved with F1 score: {best_f1:.4f}")
        
        # Plot training curve
        if epoch % 5 == 0 or epoch == config['training']['epochs']:
            plot_training_curve(
                train_losses=train_losses,
                val_losses=val_losses,
                metrics=metrics,
                output_dir=output_dir,
                epoch=epoch
            )
        
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
        'best_f1': best_f1,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics
    }

def plot_training_curve(train_losses, val_losses, metrics, output_dir, epoch):
    """Plot training curves"""
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot metrics
    for metric, values in metrics.items():
        ax2.plot(epochs, values, label=metric.capitalize())
    
    ax2.set_title('Validation Metrics')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Metrics')
    ax2.legend()
    ax2.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'training_curve_epoch{epoch}.png'))
    plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train Receipt Text Detection Model')
    parser.add_argument('--config', type=str, default='configs/detection_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='outputs/detection',
                        help='Directory to save outputs')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory with timestamp
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config to output directory
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Train model
    train_results = train_model(config, output_dir)
    
    # Save results summary
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"Best validation loss: {train_results['best_val_loss']:.4f}\n")
        f.write(f"Best F1 score: {train_results['best_f1']:.4f}\n")
        f.write("Final metrics:\n")
        for metric, values in train_results['metrics'].items():
            f.write(f"  {metric.capitalize()}: {values[-1]:.4f}\n")

if __name__ == "__main__":
    main()