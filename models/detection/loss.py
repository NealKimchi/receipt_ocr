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
                
                # Print target boxes for debugging
                print(f"Sample {i}, Number of target boxes: {len(target_boxes)}")
                if len(target_boxes) > 0:
                    print(f"First target box: {target_boxes[0]}")
                
                target_conf = target_boxes[:, 0]
                target_boxes = target_boxes[:, 1:5]  # (N, 4) [x1, y1, x2, y2]
                
                # Explicitly ensure target boxes are normalized to [0,1]
                target_boxes = torch.clamp(target_boxes, 0.0, 1.0)
                
                # Improved sampling: Sample multiple points from each target box
                h, w = pred_conf.shape[1:]
                sampled_boxes = []
                sampled_conf = []
                
                for box_idx, box in enumerate(target_boxes):
                    x1, y1, x2, y2 = box
                    
                    # Convert to pixel coordinates
                    x1_px = int(x1 * w)
                    y1_px = int(y1 * h)
                    x2_px = int(x2 * w)
                    y2_px = int(y2 * h)
                    
                    # Ensure valid pixel ranges
                    x1_px = max(0, min(x1_px, w-1))
                    y1_px = max(0, min(y1_px, h-1))
                    x2_px = max(0, min(x2_px, w-1))
                    y2_px = max(0, min(y2_px, h-1))
                    
                    # Ensure min box size of 2x2 pixels
                    if x2_px <= x1_px:
                        x2_px = min(x1_px + 2, w-1)
                    if y2_px <= y1_px:
                        y2_px = min(y1_px + 2, h-1)
                    
                    # Sample more points: center, corners, and grid points inside box
                    sample_points = [
                        ((y1_px + y2_px) // 2, (x1_px + x2_px) // 2),  # center
                        (y1_px, x1_px),  # top-left
                        (y1_px, x2_px),  # top-right
                        (y2_px, x1_px),  # bottom-left
                        (y2_px, x2_px),  # bottom-right
                    ]
                    
                    # Add grid points inside the box
                    box_height = y2_px - y1_px
                    box_width = x2_px - x1_px
                    
                    # Add more sample points if box is large enough
                    if box_height >= 4 and box_width >= 4:
                        for y_offset in range(1, 3):
                            for x_offset in range(1, 3):
                                y = y1_px + (box_height * y_offset) // 3
                                x = x1_px + (box_width * x_offset) // 3
                                sample_points.append((y, x))
                    
                    # Filter out duplicates and out-of-bounds
                    valid_points = []
                    for y, x in sample_points:
                        if 0 <= y < h and 0 <= x < w:
                            if (y, x) not in valid_points:
                                valid_points.append((y, x))
                    
                    # Ensure we have at least one valid point
                    if not valid_points and 0 <= y1_px < h and 0 <= x1_px < w:
                        valid_points.append((y1_px, x1_px))
                    
                    # Print number of valid sample points for debugging
                    print(f"Box {box_idx}, Valid sample points: {len(valid_points)}")
                    
                    for y, x in valid_points:
                        # Extract predictions at each sampled point
                        box_pred = pred_boxes[:, y, x]
                        conf_pred = pred_conf[0, y, x]
                        
                        # Explicitly normalize predicted box coordinates
                        box_pred = torch.clamp(box_pred, 0.0, 1.0)
                        
                        # Store actual box coordinates for loss computation
                        sampled_boxes.append((box_pred, box))
                        sampled_conf.append((conf_pred, target_conf[box_idx]))
                
                # Compute box loss only if we have valid sample points
                if sampled_boxes:
                    pred_boxes_tensor = torch.stack([pred for pred, _ in sampled_boxes])
                    target_boxes_tensor = torch.stack([target.to(pred_boxes_tensor.device) 
                                                    for _, target in sampled_boxes])
                    
                    # Print some values for debugging
                    if len(pred_boxes_tensor) > 0:
                        print(f"Pred box sample: {pred_boxes_tensor[0]}")
                        print(f"Target box sample: {target_boxes_tensor[0]}")
                    
                    # Compute box loss with both IoU and L1 components
                    current_box_loss = self.box_loss(pred_boxes_tensor, target_boxes_tensor)
                    box_loss += current_box_loss
                    
                    print(f"Box loss for sample {i}: {current_box_loss.item()}")
                    
                    # Compute confidence loss
                    if sampled_conf:
                        pred_conf_tensor = torch.stack([pred for pred, _ in sampled_conf])
                        target_conf_tensor = torch.stack([torch.tensor(target, device=pred_conf_tensor.device) 
                                                      for _, target in sampled_conf])
                        confidence_loss += self.confidence_loss(pred_conf_tensor, target_conf_tensor)
        
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