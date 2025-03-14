import os
import sys
import torch
import argparse
import yaml
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_loading import ReceiptDataset, receipt_collate_fn
from models.detection.model import get_model
from models.detection.metrics import (
    text_detection_evaluate, 
    extract_boxes_from_text_map, 
    calculate_box_metrics,
    comprehensive_evaluation
)
from torch.utils.data import DataLoader

def create_test_dataloader(dataset_name, batch_size, image_size, num_workers, max_samples=None, cache_dir=None):
    """Create dataloader for test dataset"""
    test_dataset = ReceiptDataset(
        dataset_name=dataset_name,
        split='val',  # Using validation split for testing since we don't have a dedicated test split
        image_size=image_size,
        max_samples=max_samples,
        cache_dir=cache_dir
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=receipt_collate_fn,
        pin_memory=True
    )
    
    return test_loader

def visualize_detection_results(image, pred_text_map, pred_boxes, ground_truth_boxes, file_path):
    """Generate visualization of detection results"""
    # Convert tensor to numpy for visualization
    if isinstance(image, torch.Tensor):
        image_np = image.detach().cpu().permute(1, 2, 0).numpy()
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = std * image_np + mean
        image_np = np.clip(image_np, 0, 1)
    else:
        image_np = image.copy()
    
    # Convert text map to numpy if it's a tensor
    if isinstance(pred_text_map, torch.Tensor):
        pred_text_map = pred_text_map.detach().cpu().squeeze().numpy()
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Predicted text map
    axes[0, 1].imshow(pred_text_map, cmap='gray')
    axes[0, 1].set_title('Predicted Text Map')
    axes[0, 1].axis('off')
    
    # Image with predicted boxes
    pred_boxes_vis = image_np.copy()
    h, w = pred_text_map.shape if hasattr(pred_text_map, 'shape') else (image_np.shape[0], image_np.shape[1])
    
    if pred_boxes is not None and len(pred_boxes) > 0:
        for box in pred_boxes:
            # Convert normalized coordinates to pixel coordinates
            if len(box) == 4:  # [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box
                x1_px, y1_px = int(x1 * w), int(y1 * h)
                x2_px, y2_px = int(x2 * w), int(y2 * h)
                cv2.rectangle(pred_boxes_vis, (x1_px, y1_px), (x2_px, y2_px), (0, 1, 0), 2)
    
    axes[1, 0].imshow(pred_boxes_vis)
    axes[1, 0].set_title('Predicted Boxes')
    axes[1, 0].axis('off')
    
    # Image with ground truth boxes
    gt_boxes_vis = image_np.copy()
    
    if ground_truth_boxes is not None and len(ground_truth_boxes) > 0:
        for box in ground_truth_boxes:
            if isinstance(box, torch.Tensor):
                box = box.detach().cpu().numpy()
            
            # Convert normalized coordinates to pixel coordinates
            if len(box) == 4:  # [x1, y1, x2, y2] format
                x1, y1, x2, y2 = box
                x1_px, y1_px = int(x1 * w), int(y1 * h)
                x2_px, y2_px = int(x2 * w), int(y2 * h)
                cv2.rectangle(gt_boxes_vis, (x1_px, y1_px), (x2_px, y2_px), (0, 0, 1), 2)
    
    axes[1, 1].imshow(gt_boxes_vis)
    axes[1, 1].set_title('Ground Truth Boxes')
    axes[1, 1].axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()

def evaluate_model(model, test_loader, config, output_dir):
    """Evaluate model on test set"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Configuration for detection
    conf_threshold = config['inference']['confidence_threshold']
    nms_threshold = config['inference']['nms_threshold']
    iou_threshold = config['evaluation']['iou_threshold']
    
    # Create directory for visualizations
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Number of samples to visualize
    num_samples_to_visualize = 10
    samples_visualized = 0
    
    # Use comprehensive evaluation function
    print("Starting comprehensive evaluation...")
    all_metrics = comprehensive_evaluation(model, test_loader, device, config)
    
    # Generate visualizations for a subset of samples
    print("Generating visualizations...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Generating visualizations")):
            if samples_visualized >= num_samples_to_visualize:
                break
                
            images = batch['image'].to(device)
            ground_truth_boxes = batch['boxes']
            
            # Forward pass
            outputs = model(images)
            
            # Process each sample in batch
            for i in range(images.size(0)):
                if samples_visualized >= num_samples_to_visualize:
                    break
                    
                # Extract predictions
                pred_text_map = outputs['text_map'][i, 0].cpu().numpy()
                pred_confidence = outputs['confidence'][i, 0].cpu().numpy() if 'confidence' in outputs else None
                
                # Extract boxes from text map
                pred_boxes, _ = extract_boxes_from_text_map(
                    pred_text_map, 
                    pred_confidence, 
                    conf_threshold=conf_threshold,
                    nms_threshold=nms_threshold
                )
                
                # Generate visualization
                vis_path = os.path.join(vis_dir, f'sample_{batch_idx}_{i}.png')
                visualize_detection_results(
                    images[i],
                    pred_text_map,
                    pred_boxes,
                    ground_truth_boxes[i],
                    vis_path
                )
                samples_visualized += 1
    
    # Print results
    print("\n===== Evaluation Results =====")
    print("Text Map Metrics:")
    print(f"  Precision: {all_metrics['text_map_precision']:.4f}")
    print(f"  Recall: {all_metrics['text_map_recall']:.4f}")
    print(f"  F1 Score: {all_metrics['text_map_f1']:.4f}")
    print(f"  IoU: {all_metrics['text_map_iou']:.4f}")
    print("\nBox-level Metrics:")
    print(f"  Precision: {all_metrics['box_precision']:.4f}")
    print(f"  Recall: {all_metrics['box_recall']:.4f}")
    print(f"  F1 Score: {all_metrics['box_f1']:.4f}")
    
    # Save results to file
    results_path = os.path.join(output_dir, 'evaluation_results.txt')
    with open(results_path, 'w') as f:
        f.write("===== Evaluation Results =====\n")
        f.write("Text Map Metrics:\n")
        f.write(f"  Precision: {all_metrics['text_map_precision']:.4f}\n")
        f.write(f"  Recall: {all_metrics['text_map_recall']:.4f}\n")
        f.write(f"  F1 Score: {all_metrics['text_map_f1']:.4f}\n")
        f.write(f"  IoU: {all_metrics['text_map_iou']:.4f}\n")
        f.write("\nBox-level Metrics:\n")
        f.write(f"  Precision: {all_metrics['box_precision']:.4f}\n")
        f.write(f"  Recall: {all_metrics['box_recall']:.4f}\n")
        f.write(f"  F1 Score: {all_metrics['box_f1']:.4f}\n")
    
    return all_metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate Text Detection Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to saved model weights')
    args = parser.parse_args()
    
    # Hardcoded arguments
    args.config = 'configs/detection_config.yaml'
    args.output_dir = 'outputs/detection/evaluation'
    args.num_samples = 100  # Evaluate 100 samples
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'eval_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Create model
    model = get_model(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )
    
    # Load model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    # Create test dataloader
    test_loader = create_test_dataloader(
        dataset_name=config['data']['dataset_name'],
        batch_size=config['training']['batch_size'],
        image_size=tuple(config['data']['image_size']),
        num_workers=config['data']['num_workers'],
        max_samples=args.num_samples,
        cache_dir=config['data'].get('cache_dir', None)
    )
    
    # Full evaluation
    results = evaluate_model(model, test_loader, config, output_dir)
    print(f"Evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()