import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and ReLU activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DownBlock(nn.Module):
    """Downsampling block with max pooling and double convolution"""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block with transposed convolution and double convolution"""
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        # Change this line to correctly handle the concatenated channels
        self.conv1 = ConvBlock(in_channels // 2 + in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Adjust dimensions if needed
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, 
                        diff_y // 2, diff_y - diff_y // 2])
        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TextDetectionLoss(nn.Module):
    """Combined loss function for text detection task"""
    def __init__(self, text_map_weight=1.0, box_weight=5.0, confidence_weight=0.5):
        super(TextDetectionLoss, self).__init__()
        self.text_map_weight = text_map_weight
        self.box_weight = box_weight  # Increased from 1.0 to 5.0
        self.confidence_weight = confidence_weight
        
        # Loss functions
        self.text_map_loss = DiceBCELoss()
        self.box_loss = EnhancedBoxLoss()  # New improved box loss
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
                target_conf = target_boxes[:, 0]
                target_boxes = target_boxes[:, 1:]  # (N, 4) [x1, y1, x2, y2]
                
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
                    
                    # Sample points: center, corners, and midpoints of edges
                    sample_points = [
                        ((y1_px + y2_px) // 2, (x1_px + x2_px) // 2),  # center
                        (y1_px, x1_px),  # top-left
                        (y1_px, x2_px),  # top-right
                        (y2_px, x1_px),  # bottom-left
                        (y2_px, x2_px),  # bottom-right
                    ]
                    
                    # Filter out duplicates and out-of-bounds
                    valid_points = []
                    for y, x in sample_points:
                        if 0 <= y < h and 0 <= x < w:
                            if (y, x) not in valid_points:
                                valid_points.append((y, x))
                    
                    for y, x in valid_points:
                        # Extract predictions at each sampled point
                        box_pred = pred_boxes[:, y, x]
                        conf_pred = pred_conf[0, y, x]
                        
                        sampled_boxes.append((box_pred, box))
                        sampled_conf.append((conf_pred, target_conf[box_idx]))
                
                # Compute box loss only if we have valid sample points
                if sampled_boxes:
                    pred_boxes_tensor = torch.stack([pred for pred, _ in sampled_boxes])
                    target_boxes_tensor = torch.stack([torch.tensor(target, device=pred_boxes_tensor.device) 
                                                    for _, target in sampled_boxes])
                    
                    # Compute box loss with both IoU and L1 components
                    box_loss += self.box_loss(pred_boxes_tensor, target_boxes_tensor)
                    
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
        # Calculate IoU loss
        iou_loss = self.iou_loss(pred_boxes, target_boxes)
        
        # Calculate regression loss (L1)
        reg_loss = F.smooth_l1_loss(pred_boxes, target_boxes)
        
        # Combine losses
        combined_loss = iou_loss + 0.5 * reg_loss
        
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

def get_model(in_channels=3, out_channels=1):
    """Helper function to create the text detection model"""
    model = TextDetectionModel(in_channels, out_channels)
    return model