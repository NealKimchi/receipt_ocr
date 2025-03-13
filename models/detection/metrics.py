import numpy as np
import torch
import cv2

def calculate_iou(pred_mask, gt_mask):
    """Calculate Intersection over Union for binary masks"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_precision_recall_f1(pred_mask, gt_mask):
    """Calculate precision, recall and F1 score manually"""
    # True positives: predicted positive (1) and actually positive (1)
    true_positives = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    
    # False positives: predicted positive (1) but actually negative (0)
    false_positives = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
    
    # False negatives: predicted negative (0) but actually positive (1)
    false_negatives = np.logical_and(pred_mask == 0, gt_mask == 1).sum()
    
    # Calculate precision: TP / (TP + FP)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    
    # Calculate recall: TP / (TP + FN)
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    
    # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def calculate_detection_metrics(pred_mask, gt_mask):
    """Calculate precision, recall, F1, and IoU for text detection"""
    # Calculate precision, recall, F1
    precision, recall, f1 = calculate_precision_recall_f1(pred_mask, gt_mask)
    
    # Calculate IoU
    iou = calculate_iou(pred_mask, gt_mask)
    
    return precision, recall, f1, iou

def text_detection_evaluate(model, val_loader, device, threshold=0.5):
    """Evaluate text detection model on validation set"""
    model.eval()
    
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_iou = 0
    valid_samples = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            images = batch['image'].to(device)
            gt_maps = batch['text_map'].cpu().numpy()
            
            # Get predictions
            outputs = model(images)
            pred_maps = outputs['text_map'].cpu().numpy()
            
            # Calculate metrics for each sample
            batch_size = pred_maps.shape[0]
            for i in range(batch_size):
                pred_map = pred_maps[i, 0]
                gt_map = gt_maps[i, 0]
                
                # Binarize prediction map
                pred_binary = (pred_map > threshold).astype(np.uint8)
                gt_binary = (gt_map > 0.5).astype(np.uint8)
                
                # Skip samples with no text regions
                if np.sum(gt_binary) == 0:
                    continue
                
                # Calculate metrics
                precision, recall, f1, iou = calculate_detection_metrics(pred_binary, gt_binary)
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_iou += iou
                valid_samples += 1
    
    # Average metrics
    if valid_samples > 0:
        avg_precision = total_precision / valid_samples
        avg_recall = total_recall / valid_samples
        avg_f1 = total_f1 / valid_samples
        avg_iou = total_iou / valid_samples
    else:
        avg_precision = 0
        avg_recall = 0
        avg_f1 = 0
        avg_iou = 0
    
    metrics = {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'iou': avg_iou
    }
    
    return metrics