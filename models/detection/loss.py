import torch
import torch.nn as nn
import torch.nn.functional as F


class TextDetectionLoss(nn.Module):
    """Combined loss function for text detection task"""
    def __init__(self, text_map_weight=1.0, box_weight=20.0, confidence_weight=0.5):
        super(TextDetectionLoss, self).__init__()
        self.text_map_weight = text_map_weight
        self.box_weight = box_weight  # Increased to 20.0
        self.confidence_weight = confidence_weight
        
        # Loss functions
        self.text_map_loss = DiceBCELoss()
        self.box_loss = EnhancedBoxLoss()
        self.confidence_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions (dict): Model predictions containing 'text_map', 'confidence', 'bbox_coords'
            targets (dict): Target values containing 'text_map', 'boxes'
        
        Returns:
            dict: Loss values including 'total_loss', 'text_map_loss', 'box_loss', 'confidence_loss'
        """
        # Text map segmentation loss
        text_map_loss = self.text_map_loss(predictions['text_map'], targets['text_map'])
        
        # Initialize box and confidence losses
        box_loss = torch.tensor(0.0, device=text_map_loss.device)
        confidence_loss = torch.tensor(0.0, device=text_map_loss.device)
        
        batch_size = predictions['text_map'].size(0)
        valid_samples = 0
        total_boxes = 0
        
        # Process each item in the batch individually due to variable box counts
        for i in range(batch_size):
            pred_conf = predictions['confidence'][i]  # (1, H, W)
            pred_boxes = predictions['bbox_coords'][i]  # (4, H, W)
            
            # Print prediction information
            print(f"Sample {i}, Predicted bbox_coords shape: {pred_boxes.shape}")
            
            # Get image dimensions for normalization
            h, w = pred_conf.shape[1:]
            
            # Check if we have target boxes for this sample
            if 'boxes' in targets and i < len(targets['boxes']) and targets['boxes'][i] is not None:
                # Get the boxes for this sample
                sample_boxes = targets['boxes'][i]
                
                if len(sample_boxes) > 0:
                    valid_samples += 1
                    total_boxes += len(sample_boxes)
                    
                    # Process boxes
                    target_conf_list = []
                    target_boxes_list = []
                    
                    for box in sample_boxes:
                        # Extract confidence and coordinates
                        conf = box[0]
                        x1, y1, x2, y2 = box[1:]
                        
                        # Normalize coordinates to [0,1]
                        norm_x1 = x1 / w
                        norm_y1 = y1 / h
                        norm_x2 = x2 / w
                        norm_y2 = y2 / h
                        
                        # Clamp values to ensure proper box format (0≤x1<x2≤1, 0≤y1<y2≤1)
                        norm_x1 = max(0.0, min(0.999, norm_x1))
                        norm_y1 = max(0.0, min(0.999, norm_y1))
                        norm_x2 = max(norm_x1 + 0.001, min(1.0, norm_x2))
                        norm_y2 = max(norm_y1 + 0.001, min(1.0, norm_y2))
                        
                        target_conf_list.append(conf)
                        target_boxes_list.append([norm_x1, norm_y1, norm_x2, norm_y2])
                    
                    # Create tensor for normalized boxes
                    if len(target_boxes_list) > 0:
                        target_boxes = torch.tensor(target_boxes_list, device=text_map_loss.device)
                        target_conf = torch.tensor(target_conf_list, device=text_map_loss.device)
                        
                        print(f"Sample {i}, First normalized box: {target_boxes_list[0]}")
                        
                        # Sample points from each box for loss calculation
                        sampled_boxes = []
                        sampled_conf = []
                        
                        for box_idx, box in enumerate(target_boxes):
                            x1, y1, x2, y2 = box
                            
                            # Convert normalized coordinates to pixel coordinates for sampling
                            x1_px = int(x1.item() * w)
                            y1_px = int(y1.item() * h)
                            x2_px = int(x2.item() * w)
                            y2_px = int(y2.item() * h)
                            
                            # Ensure valid pixel ranges
                            x1_px = max(0, min(x1_px, w-1))
                            y1_px = max(0, min(y1_px, h-1))
                            x2_px = max(x1_px + 1, min(x2_px, w-1))
                            y2_px = max(y1_px + 1, min(y2_px, h-1))
                            
                            # Sample points: center, corners, grid points
                            sample_points = []
                            
                            # Center point
                            center_y = (y1_px + y2_px) // 2
                            center_x = (x1_px + x2_px) // 2
                            sample_points.append((center_y, center_x))
                            
                            # Corners
                            sample_points.append((y1_px, x1_px))  # Top-left
                            sample_points.append((y1_px, x2_px))  # Top-right
                            sample_points.append((y2_px, x1_px))  # Bottom-left
                            sample_points.append((y2_px, x2_px))  # Bottom-right
                            
                            # Filter out-of-bounds points
                            valid_points = []
                            for y, x in sample_points:
                                if 0 <= y < h and 0 <= x < w:
                                    valid_points.append((y, x))
                            
                            print(f"Box {box_idx}, Valid sample points: {len(valid_points)}")
                            
                            # Extract predictions at each valid point
                            for y, x in valid_points:
                                box_pred = pred_boxes[:, y, x]
                                conf_pred = pred_conf[0, y, x]
                                
                                sampled_boxes.append((box_pred, box))
                                sampled_conf.append((conf_pred, target_conf[box_idx]))
                        
                        # Compute box loss if we have valid sample points
                        if sampled_boxes:
                            pred_boxes_tensor = torch.stack([pred for pred, _ in sampled_boxes])
                            target_boxes_tensor = torch.stack([target.to(pred_boxes_tensor.device) for _, target in sampled_boxes])
                            
                            # Print some sample values
                            if len(pred_boxes_tensor) > 0:
                                print(f"Pred box sample: {pred_boxes_tensor[0]}")
                                print(f"Target box sample: {target_boxes_tensor[0]}")
                            
                            # Compute individual loss components
                            iou_loss = self.box_loss.iou_loss(pred_boxes_tensor, target_boxes_tensor)
                            l1_loss = F.smooth_l1_loss(pred_boxes_tensor, target_boxes_tensor)
                            giou_loss = self.box_loss.giou_loss(pred_boxes_tensor, target_boxes_tensor)
                            
                            # Print individual loss components
                            print(f"IoU Loss: {iou_loss.item()}, L1 Loss: {l1_loss.item()}, GIoU Loss: {giou_loss.item()}")
                            
                            current_box_loss = self.box_loss(pred_boxes_tensor, target_boxes_tensor)
                            box_loss += current_box_loss
                            
                            print(f"Box loss for sample {i}: {current_box_loss.item()}")
                            
                            # Compute confidence loss
                            if sampled_conf:
                                pred_conf_tensor = torch.stack([pred for pred, _ in sampled_conf])
                                target_conf_tensor = torch.stack([target.to(pred_conf_tensor.device) for _, target in sampled_conf])
                                confidence_loss += self.confidence_loss(pred_conf_tensor, target_conf_tensor)
        
        # Print summary statistics
        print(f"Total valid samples: {valid_samples}, Total boxes: {total_boxes}")
        print(f"Raw box loss (before averaging): {box_loss.item()}")
        
        # Average losses over valid samples
        if valid_samples > 0:
            box_loss /= valid_samples
            confidence_loss /= valid_samples
        
        # Print final losses
        print(f"Final box loss (after averaging): {box_loss.item()}")
        print(f"Text map loss: {text_map_loss.item()}, Confidence loss: {confidence_loss.item()}")
        
        # Combine losses
        total_loss = (
            self.text_map_weight * text_map_loss +
            self.box_weight * box_loss +
            self.confidence_weight * confidence_loss
        )
        
        return {
            'total_loss': total_loss,
            'text_map_loss': text_map_loss,
            'box_loss': box_loss,
            'confidence_loss': confidence_loss
        }
        
class DiceBCELoss(nn.Module):
    """Dice and BCE combined loss for segmentation tasks"""
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCELoss(weight=weight, size_average=size_average)
    
    def forward(self, inputs, targets):
        # BCE Loss
        bce_loss = self.bce(inputs, targets)
        
        # Dice Loss
        smooth = 1e-5
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        dice_loss = 1 - dice_score
        
        # Combine losses
        return 0.5 * bce_loss + 0.5 * dice_loss


class IoULoss(nn.Module):
    """IoU-based loss for bounding box regression"""
    def __init__(self):
        super(IoULoss, self).__init__()
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (N, 4) [x1, y1, x2, y2]
            target_boxes: (N, 4) [x1, y1, x2, y2]
        """
        # Ensure both are in the same format
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        
        target_x1 = target_boxes[:, 0]
        target_y1 = target_boxes[:, 1]
        target_x2 = target_boxes[:, 2]
        target_y2 = target_boxes[:, 3]
        
        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Calculate intersection
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)
        
        # Intersection height and width
        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)
        
        # Intersection area
        inter = w * h
        
        # Calculate union
        union = pred_area + target_area - inter
        
        # Calculate IoU
        iou = inter / (union + 1e-6)
        
        # IoU loss
        iou_loss = 1 - iou.mean()
        
        return iou_loss
class EnhancedBoxLoss(nn.Module):
    """Enhanced IoU-based loss for bounding box regression"""
    def __init__(self):
        super(EnhancedBoxLoss, self).__init__()
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: (N, 4) [x1, y1, x2, y2]
            target_boxes: (N, 4) [x1, y1, x2, y2]
        """
        # Ensure valid coordinate ordering (x1 < x2, y1 < y2)
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        
        # Correct any invalid coordinates (ensure x1 < x2, y1 < y2)
        pred_x1 = torch.clamp(pred_x1, 0.0, 0.99)
        pred_y1 = torch.clamp(pred_y1, 0.0, 0.99)
        pred_x2 = torch.clamp(pred_x2, 0.01, 1.0)
        pred_y2 = torch.clamp(pred_y2, 0.01, 1.0)
        
        pred_x1, pred_x2 = torch.min(pred_x1, pred_x2), torch.max(pred_x1, pred_x2)
        pred_y1, pred_y2 = torch.min(pred_y1, pred_y2), torch.max(pred_y1, pred_y2)
        
        # Create corrected predicted boxes
        corrected_pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
        
        # Calculate IoU loss
        iou_loss = self.iou_loss(corrected_pred_boxes, target_boxes)
        
        # Calculate regression loss (L1)
        reg_loss = F.smooth_l1_loss(corrected_pred_boxes, target_boxes)
        
        # Calculate GIoU loss for better gradients
        giou_loss = self.giou_loss(corrected_pred_boxes, target_boxes)
        
        # Combine losses
        combined_loss = 0.5 * iou_loss + 0.3 * reg_loss + 0.2 * giou_loss
        
        return combined_loss
    
    def iou_loss(self, pred_boxes, target_boxes):
        # Ensure both are in the same format
        pred_x1 = pred_boxes[:, 0]
        pred_y1 = pred_boxes[:, 1]
        pred_x2 = pred_boxes[:, 2]
        pred_y2 = pred_boxes[:, 3]
        
        target_x1 = target_boxes[:, 0]
        target_y1 = target_boxes[:, 1]
        target_x2 = target_boxes[:, 2]
        target_y2 = target_boxes[:, 3]
        
        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Calculate intersection
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)
        
        # Intersection height and width
        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)
        
        # Intersection area
        inter = w * h
        
        # Calculate union
        union = pred_area + target_area - inter
        
        # Calculate IoU
        iou = inter / (union + 1e-6)
        
        # Use log-based IoU loss for better gradients
        iou_loss = -torch.log(iou + 1e-6).mean()
        
        return iou_loss
    
    def giou_loss(self, pred_boxes, target_boxes):
        # Extract coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(1)
        
        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Calculate intersection
        x1 = torch.max(pred_x1, target_x1)
        y1 = torch.max(pred_y1, target_y1)
        x2 = torch.min(pred_x2, target_x2)
        y2 = torch.min(pred_y2, target_y2)
        
        # Intersection height and width
        w = torch.clamp(x2 - x1, min=0)
        h = torch.clamp(y2 - y1, min=0)
        
        # Intersection area
        inter = w * h
        
        # Union area
        union = pred_area + target_area - inter
        
        # IoU
        iou = inter / (union + 1e-6)
        
        # Enclosing box
        enclosing_x1 = torch.min(pred_x1, target_x1)
        enclosing_y1 = torch.min(pred_y1, target_y1)
        enclosing_x2 = torch.max(pred_x2, target_x2)
        enclosing_y2 = torch.max(pred_y2, target_y2)
        
        # Area of enclosing box
        enclosing_area = (enclosing_x2 - enclosing_x1) * (enclosing_y2 - enclosing_y1)
        
        # GIoU
        giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
        
        # GIoU loss
        giou_loss = (1 - giou).mean()
        
        return giou_loss
    
def get_loss_function(text_map_weight=1.0, box_weight=20.0, confidence_weight=0.5):
    """Helper function to create the loss function"""
    return TextDetectionLoss(
        text_map_weight=text_map_weight,
        box_weight=box_weight,
        confidence_weight=confidence_weight
    )