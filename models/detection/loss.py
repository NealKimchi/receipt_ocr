import torch
import torch.nn as nn
import torch.nn.functional as F


class TextDetectionLoss(nn.Module):
    """Combined loss function for text detection task"""
    def __init__(self, text_map_weight=1.0, box_weight=1.0, confidence_weight=0.5):
        super(TextDetectionLoss, self).__init__()
        self.text_map_weight = text_map_weight
        self.box_weight = box_weight
        self.confidence_weight = confidence_weight
        
        # Loss functions
        self.text_map_loss = DiceBCELoss()
        self.box_loss = IoULoss()
        self.confidence_loss = nn.BCELoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions (dict): Model predictions containing 'text_map', 'confidence', 'bbox_coords'
            targets (dict): Target values containing 'text_map', 'boxes'
        
        Returns:
            float: Combined loss value
        """
        # Text map segmentation loss
        text_map_loss = self.text_map_loss(predictions['text_map'], targets['text_map'])
        
        # Initialize box and confidence losses
        box_loss = torch.tensor(0.0, device=text_map_loss.device)
        confidence_loss = torch.tensor(0.0, device=text_map_loss.device)
        
        batch_size = predictions['text_map'].size(0)
        valid_samples = 0
        
        # Process each item in the batch individually due to variable box counts
        for i in range(batch_size):
            pred_conf = predictions['confidence'][i]  # (1, H, W)
            pred_boxes = predictions['bbox_coords'][i]  # (4, H, W)
            
            # Only compute box loss if there are target boxes
            if len(targets['boxes'][i]) > 0:
                valid_samples += 1
                
                # Extract target boxes and confidences
                target_boxes = targets['boxes'][i]  # (N, 5) [conf, x1, y1, x2, y2]
                target_conf = target_boxes[:, 0]
                target_boxes = target_boxes[:, 1:]  # (N, 4) [x1, y1, x2, y2]
                
                # Sample points from predicted boxes at target locations
                h, w = pred_conf.shape[1:]
                
                # Convert target box centers to indices in the prediction map
                box_centers_y = ((target_boxes[:, 1] + target_boxes[:, 3]) / 2 * h).long().clamp(0, h-1)
                box_centers_x = ((target_boxes[:, 0] + target_boxes[:, 2]) / 2 * w).long().clamp(0, w-1)
                
                # Extract predictions at target box centers
                sampled_conf = pred_conf[0, box_centers_y, box_centers_x]
                
                # Extract box predictions at target centers
                # pred_boxes: (4, H, W)
                sampled_boxes = torch.stack([
                    pred_boxes[j, box_centers_y, box_centers_x] for j in range(4)
                ], dim=1)  # (N, 4)
                
                # Compute box loss
                box_loss += self.box_loss(sampled_boxes, target_boxes)
                
                # Compute confidence loss
                confidence_loss += self.confidence_loss(sampled_conf, target_conf)
        
        # Average losses over valid samples
        if valid_samples > 0:
            box_loss /= valid_samples
            confidence_loss /= valid_samples
        
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


def get_loss_function(text_map_weight=1.0, box_weight=1.0, confidence_weight=0.5):
    """Helper function to create the loss function"""
    return TextDetectionLoss(
        text_map_weight=text_map_weight,
        box_weight=box_weight,
        confidence_weight=confidence_weight
    )