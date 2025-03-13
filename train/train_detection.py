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
    binary_map = (text_map_pred > 0.5).astype(np.uint8)
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

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, output_dir):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    batch_losses = []
    
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
        optimizer.step()
        
        # Accumulate loss
        epoch_loss += total_loss.item()
        batch_losses.append(total_loss.item())
        
        # Print detailed loss components periodically
        log_interval = config['training'].get('log_interval', 10)
        if (batch_idx + 1) % log_interval == 0:
            print(f"\nBatch {batch_idx+1}/{len(train_loader)}: "
                  f"Total Loss: {total_loss.item():.4f}, "
                  f"Text Map Loss: {loss_dict['text_map_loss'].item():.4f}, "
                  f"Box Loss: {loss_dict['box_loss'].item():.4f}, "
                  f"Confidence Loss: {loss_dict['confidence_loss'].item():.4f}")
        
        # Visualize predictions periodically
        vis_interval = config['training'].get('visualization_interval', 50)
        if (batch_idx + 1) % vis_interval == 0:
            vis_path = os.path.join(output_dir, f"train_vis_epoch{epoch}_batch{batch_idx}.png")
            visualize_predictions(images[0], outputs, batch, 0, device, save_path=vis_path)
    
    # Average training loss for this epoch
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Print epoch summary
    elapsed_time = time.time() - start_time
    print(f"Epoch {epoch} completed in {elapsed_time:.2f}s - Train Loss: {avg_train_loss:.4f}")
    
    return avg_train_loss, batch_losses

def validate(model, val_loader, criterion, device, epoch, config, output_dir):
    """Validate the model"""
    model.eval()
    val_loss = 0
    metrics = {}
    
    # Initialize metrics
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0
    valid_samples = 0
    
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
            for i in range(len(images)):
                # Calculate metrics for each image in batch
                pred_map = outputs['text_map'][i, 0].cpu().numpy()
                gt_map = text_maps[i, 0].cpu().numpy()
                
                # Threshold prediction map
                pred_binary = (pred_map > 0.5).astype(np.uint8)
                gt_binary = (gt_map > 0.5).astype(np.uint8)
                
                # Only calculate metrics if there are ground truth boxes
                if np.sum(gt_binary) > 0:
                    # Calculate precision, recall, F1, IoU
                    precision, recall, f1, iou = calculate_detection_metrics(
                        pred_binary, gt_binary
                    )
                    
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1
                    total_iou += iou
                    valid_samples += 1
            
            # Visualize predictions on first validation batch
            if batch_idx == 0:
                vis_path = os.path.join(output_dir, f"val_vis_epoch{epoch}.png")
                visualize_predictions(images[0], outputs, batch, 0, device, save_path=vis_path)
    
    # Average validation loss
    avg_val_loss = val_loss / len(val_loader)
    
    # Calculate average metrics
    if valid_samples > 0:
        metrics['precision'] = total_precision / valid_samples
        metrics['recall'] = total_recall / valid_samples
        metrics['f1'] = total_f1 / valid_samples
        metrics['iou'] = total_iou / valid_samples
    else:
        metrics['precision'] = 0
        metrics['recall'] = 0
        metrics['f1'] = 0
        metrics['iou'] = 0
    
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
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Load data
    train_loader, val_loader = get_data_loaders(
        dataset_name=config['data'].get('dataset_name', "mychen76/invoices-and-receipts_ocr_v2"),
        batch_size=config['training'].get('batch_size', 8),
        image_size=tuple(config['data'].get('image_size', (512, 512))),
        num_workers=config['training'].get('num_workers', 4),
        max_samples=config['data'].get('max_samples', None)
    )
    
    # Create model
    model = get_model(
        in_channels=config['model'].get('in_channels', 3),
        out_channels=config['model'].get('out_channels', 1)
    )
    model = model.to(device)
    
    # Create loss function
    criterion = get_loss_function(
        text_map_weight=config['loss'].get('text_map_weight', 1.0),
        box_weight=config['loss'].get('box_weight', 10.0),
        confidence_weight=config['loss'].get('confidence_weight', 0.5)
    )
    
    # Create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training'].get('learning_rate', 0.001),
        weight_decay=config['training'].get('weight_decay', 0.0001)
    )
    
    # Create learning rate scheduler
    scheduler = None
    if config['training'].get('use_scheduler', True):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training'].get('scheduler_factor', 0.5),
            patience=config['training'].get('scheduler_patience', 5),
            verbose=True
        )
    
    # Training loop variables
    num_epochs = config['training'].get('num_epochs', 50)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    metrics_history = {'precision': [], 'recall': [], 'f1': [], 'iou': []}
    batch_loss_history = []
    early_stopping_counter = 0
    early_stopping_patience = config['training'].get('early_stopping_patience', 10)
    
    # Training loop
    for epoch in range(num_epochs):
        # Train for one epoch
        avg_train_loss, batch_losses = train_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            config=config,
            output_dir=output_dir
        )
        
        train_losses.append(avg_train_loss)
        batch_loss_history.extend(batch_losses)
        
        # Validate the model
        avg_val_loss, metrics = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch + 1,
            config=config,
            output_dir=output_dir
        )
        
        val_losses.append(avg_val_loss)
        
        # Update metrics history
        for metric, value in metrics.items():
            metrics_history[metric].append(value)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'metrics': metrics
        }
        
        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth'))
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"New best model saved!")
        
        # Update learning rate scheduler
        if scheduler is not None:
            scheduler.step(avg_val_loss)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'final_model.pth'))
    
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'))
    
    # Plot metrics
    plt.figure(figsize=(12, 8))
    for i, (metric, values) in enumerate(metrics_history.items(), 1):
        plt.subplot(2, 2, i)
        plt.plot(values)
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs Epoch')
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.png'))
    
    # Plot batch loss
    plt.figure(figsize=(10, 5))
    plt.plot(batch_loss_history)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Batch Loss')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'batch_loss.png'))
    
    # Print training summary
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final metrics:")
    for metric, values in metrics_history.items():
        print(f"  {metric.capitalize()}: {values[-1]:.4f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'metrics': metrics_history,
        'best_val_loss': best_val_loss
    }

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
    
    # Train model
    train_results = train_model(config, output_dir)
    
    # Save results summary
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"Best validation loss: {train_results['best_val_loss']:.4f}\n")
        f.write("Final metrics:\n")
        for metric, values in train_results['metrics'].items():
            f.write(f"  {metric.capitalize()}: {values[-1]:.4f}\n")


if __name__ == "__main__":
    main()