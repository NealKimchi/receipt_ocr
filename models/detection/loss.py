import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for text detection map
    Better handles class imbalance than BCE
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss to focus on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class GIoULoss(nn.Module):
    """
    Generalized IoU Loss for bounding box regression
    Better than L1/L2 for box regression
    """
    def __init__(self):
        super(GIoULoss, self).__init__()
    
    def forward(self, pred, target):
        # pred and target should be in [x, y, w, h] format
        # Convert to [x1, y1, x2, y2] format
        pred_x1, pred_y1 = pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2
        pred_x2, pred_y2 = pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
        
        target_x1, target_y1 = target[:, 0] - target[:, 2] / 2, target[:, 1] - target[:, 3] / 2
        target_x2, target_y2 = target[:, 0] + target[:, 2] / 2, target[:, 1] + target[:, 3] / 2
        
        # Intersection
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        
        inter_area = inter_w * inter_h
        
        # Union
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-8)
        
        # Smallest enclosing box
        enclosing_x1 = torch.min(pred_x1, target_x1)
        enclosing_y1 = torch.min(pred_y1, target_y1)
        enclosing_x2 = torch.max(pred_x2, target_x2)
        enclosing_y2 = torch.max(pred_y2, target_y2)
        
        enclosing_w = (enclosing_x2 - enclosing_x1).clamp(min=0)
        enclosing_h = (enclosing_y2 - enclosing_y1).clamp(min=0)
        
        enclosing_area = enclosing_w * enclosing_h
        
        # GIoU
        giou = iou - (enclosing_area - union_area) / (enclosing_area + 1e-8)
        
        return (1 - giou).mean()


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


class TextDetectionLoss(nn.Module):
    def __init__(self, text_map_weight=1.0, box_weight=0.1, confidence_weight=0.5):
        super(TextDetectionLoss, self).__init__()
        self.text_map_weight = text_map_weight
        self.box_weight = box_weight
        self.confidence_weight = confidence_weight
        
        # Use better losses for each component
        self.text_map_loss = DiceLoss()  # Better for segmentation than BCE
        self.box_loss = GIoULoss()       # Better for box regression
        self.confidence_loss = FocalLoss(alpha=0.5, gamma=2.0)  # Better for imbalanced data
    
    def forward(self, predictions, targets):
        # Extract predictions
        pred_text_map = predictions['text_map']
        pred_confidence = predictions['confidence']
        pred_boxes = predictions['bbox_coords']
        
        # Extract targets
        target_text_map = targets['text_map']
        
        # Create target confidence map (same as text map for now)
        target_confidence = target_text_map.clone()
        
        # Calculate text map loss
        text_map_loss = self.text_map_loss(pred_text_map, target_text_map)
        
        # Calculate confidence loss
        confidence_loss = self.confidence_loss(pred_confidence, target_confidence)
        
        # Initialize box loss
        box_loss = torch.tensor(0.0, device=pred_text_map.device)
        
        # Process boxes - only if target boxes exist
        if 'boxes' in targets and targets['boxes']:
            # Each image in batch has variable number of boxes
            batch_size = pred_boxes.size(0)
            
            # Process each image in batch
            for i in range(batch_size):
                image_target_boxes = targets['boxes'][i]
                
                # Skip if no target boxes
                if not image_target_boxes or len(image_target_boxes) == 0:
                    continue
                
                # Convert to tensor if it's a list
                if isinstance(image_target_boxes, list):
                    # Convert to tensor and move to right device
                    image_target_boxes = torch.tensor(image_target_boxes, 
                                                    device=pred_boxes.device, 
                                                    dtype=torch.float32)
                
                # Get corresponding predicted boxes - take from regions where text exists
                # Create mask from text map
                mask = (target_text_map[i] > 0.5).squeeze()
                
                # Skip if mask is empty
                if mask.sum() == 0:
                    continue
                
                # Extract box predictions based on mask
                # First flatten the spatial dimensions
                h, w = mask.size()
                flat_mask = mask.view(-1)  # [H*W]
                flat_pred_boxes = pred_boxes[i].permute(1, 2, 0).view(-1, 4)  # [H*W, 4]
                
                # Get boxes where mask is True
                masked_pred_boxes = flat_pred_boxes[flat_mask]
                
                # If we have at least one positive prediction, calculate box loss
                if masked_pred_boxes.size(0) > 0:
                    # For simplicity, use the first prediction for each target box
                    # A more advanced approach could use matching
                    used_pred_boxes = masked_pred_boxes[:min(len(image_target_boxes), 
                                                           masked_pred_boxes.size(0))]
                    
                    used_target_boxes = image_target_boxes[:min(len(image_target_boxes),
                                                              masked_pred_boxes.size(0))]
                    
                    # Calculate box loss
                    img_box_loss = self.box_loss(used_pred_boxes, used_target_boxes)
                    box_loss = box_loss + img_box_loss
            
            # Average box loss over batch
            if batch_size > 0:
                box_loss = box_loss / batch_size
        
        # Combine losses with weights
        total_loss = (self.text_map_weight * text_map_loss + 
                     self.box_weight * box_loss + 
                     self.confidence_weight * confidence_loss)
        
        # Create loss dict for logging
        loss_dict = {
            'total_loss': total_loss,
            'text_map_loss': text_map_loss,
            'box_loss': box_loss,
            'confidence_loss': confidence_loss
        }
        
        return loss_dict


def get_loss_function(text_map_weight=1.0, box_weight=0.1, confidence_weight=0.5):
    """Helper function to create the loss function"""
    return TextDetectionLoss(
        text_map_weight=text_map_weight,
        box_weight=box_weight,
        confidence_weight=confidence_weight
    )