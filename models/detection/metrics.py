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

# New additions from the enhanced metrics module

def calculate_box_metrics(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Calculate precision, recall, and F1 score for box predictions
    
    Args:
        pred_boxes: List of predicted boxes in format [x1, y1, x2, y2]
        gt_boxes: List of ground truth boxes in format [x1, y1, x2, y2]
        iou_threshold: IoU threshold for considering a detection as correct
        
    Returns:
        precision, recall, f1: Box-level metrics
    """
    if not pred_boxes or not gt_boxes:
        if not pred_boxes and not gt_boxes:
            return 1.0, 1.0, 1.0  # Both empty - perfect match
        elif not pred_boxes:
            return 0.0, 0.0, 0.0  # No predictions but have ground truth - zero recall
        else:
            return 0.0, 0.0, 0.0  # Predictions but no ground truth - zero precision
    
    # Convert lists to numpy arrays if they aren't already
    if isinstance(pred_boxes, list):
        pred_boxes = np.array(pred_boxes)
    if isinstance(gt_boxes, list):
        gt_boxes = np.array(gt_boxes)
    
    # Calculate IoU for all combinations of predicted and ground truth boxes
    ious = np.zeros((len(pred_boxes), len(gt_boxes)))
    
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            # Calculate intersection
            x1 = max(pred_box[0], gt_box[0])
            y1 = max(pred_box[1], gt_box[1])
            x2 = min(pred_box[2], gt_box[2])
            y2 = min(pred_box[3], gt_box[3])
            
            # Calculate areas
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            inter = w * h
            
            pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
            
            union = pred_area + gt_area - inter
            ious[i, j] = inter / (union + 1e-8)
    
    # Match predictions to ground truth
    num_matched_gt = 0
    matched_gt = set()
    
    for i in range(len(pred_boxes)):
        best_iou = ious[i].max() if len(ious[i]) > 0 else 0
        if best_iou >= iou_threshold:
            best_gt_idx = ious[i].argmax()
            if best_gt_idx not in matched_gt:
                num_matched_gt += 1
                matched_gt.add(best_gt_idx)
    
    # Calculate metrics
    precision = num_matched_gt / len(pred_boxes) if len(pred_boxes) > 0 else 0.0
    recall = num_matched_gt / len(gt_boxes) if len(gt_boxes) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

def extract_boxes_from_text_map(text_map, confidence_map=None, conf_threshold=0.3, nms_threshold=0.3):
    """
    Extract bounding boxes from predicted text map
    
    Args:
        text_map: Predicted text map as numpy array
        confidence_map: Optional confidence map for scoring boxes
        conf_threshold: Threshold for binarizing text map
        nms_threshold: Non-maximum suppression threshold
        
    Returns:
        boxes: List of boxes in format [x1, y1, x2, y2]
        scores: List of confidence scores for each box
    """
    # Threshold text map
    binary_map = (text_map > conf_threshold).astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract boxes
    boxes = []
    scores = []
    h, w = text_map.shape
    
    for contour in contours:
        if cv2.contourArea(contour) > 10:  # Filter small noise regions
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            # Convert to normalized coordinates
            x1, y1 = x / w, y / h
            x2, y2 = (x + w_box) / w, (y + h_box) / h
            
            # Ensure coordinates are within bounds
            x1 = max(0.0, min(1.0, x1))
            y1 = max(0.0, min(1.0, y1))
            x2 = max(0.0, min(1.0, x2))
            y2 = max(0.0, min(1.0, y2))
            
            # Calculate confidence score if confidence map is provided
            if confidence_map is not None:
                region = confidence_map[y:y+h_box, x:x+w_box]
                score = np.mean(region) if region.size > 0 else 0.0
            else:
                # Use mean value from text map
                region = text_map[y:y+h_box, x:x+w_box]
                score = np.mean(region) if region.size > 0 else 0.0
            
            boxes.append([x1, y1, x2, y2])
            scores.append(score)
    
    # Apply non-maximum suppression if we have more than one box
    if len(boxes) > 1 and nms_threshold < 1.0:
        boxes = np.array(boxes)
        scores = np.array(scores)
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by score
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
            
            w_inter = np.maximum(0.0, xx2 - xx1)
            h_inter = np.maximum(0.0, yy2 - yy1)
            inter = w_inter * h_inter
            
            # IoU = inter / (areas[i] + areas[order[1:]] - inter)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-8)
            
            # Keep only boxes with IoU less than threshold
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]
        
        boxes = boxes[keep].tolist()
        scores = scores[keep].tolist()
    
    return boxes, scores

def evaluate_boxes(model, test_loader, device, conf_threshold=0.3, nms_threshold=0.3, iou_threshold=0.5):
    """
    Evaluate model's bounding box predictions
    
    Args:
        model: The detection model
        test_loader: DataLoader for test data
        device: Device to run inference on
        conf_threshold: Confidence threshold for detection
        nms_threshold: NMS threshold for box filtering
        iou_threshold: IoU threshold for box matching
        
    Returns:
        dict: Dictionary with box-level metrics
    """
    model.eval()
    
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            gt_boxes_batch = batch['boxes']
            
            # Forward pass
            outputs = model(images)
            
            # Process each sample in batch
            for i in range(images.size(0)):
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
                
                # Calculate box metrics
                gt_boxes = gt_boxes_batch[i]
                precision, recall, f1 = calculate_box_metrics(pred_boxes, gt_boxes, iou_threshold)
                
                total_precision += precision
                total_recall += recall
                total_f1 += f1
                total_samples += 1
    
    # Calculate average metrics
    avg_precision = total_precision / total_samples if total_samples > 0 else 0.0
    avg_recall = total_recall / total_samples if total_samples > 0 else 0.0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0.0
    
    return {
        'box_precision': avg_precision,
        'box_recall': avg_recall,
        'box_f1': avg_f1
    }

def comprehensive_evaluation(model, test_loader, device, config):
    """
    Perform comprehensive evaluation of the text detection model
    
    Args:
        model: The detection model
        test_loader: DataLoader for test data
        device: Device to run inference on
        config: Configuration dictionary with evaluation parameters
        
    Returns:
        dict: Dictionary with all evaluation metrics
    """
    model.eval()
    
    # Get configuration parameters
    conf_threshold = config['inference']['confidence_threshold']
    nms_threshold = config['inference']['nms_threshold']
    iou_threshold = config['evaluation']['iou_threshold']
    
    # Get text map metrics (using existing function)
    text_map_metrics = text_detection_evaluate(model, test_loader, device, threshold=conf_threshold)
    
    # Get box metrics
    box_metrics = evaluate_boxes(model, test_loader, device, 
                                conf_threshold=conf_threshold,
                                nms_threshold=nms_threshold,
                                iou_threshold=iou_threshold)
    
    # Combine metrics
    all_metrics = {
        'text_map_precision': text_map_metrics['precision'],
        'text_map_recall': text_map_metrics['recall'],
        'text_map_f1': text_map_metrics['f1'],
        'text_map_iou': text_map_metrics['iou'],
        'box_precision': box_metrics['box_precision'],
        'box_recall': box_metrics['box_recall'],
        'box_f1': box_metrics['box_f1']
    }
    
    return all_metrics