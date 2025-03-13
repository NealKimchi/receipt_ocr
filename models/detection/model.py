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


class TextDetectionModel(nn.Module):
    """U-Net inspired architecture for text detection in receipts"""
    def __init__(self, in_channels=3, out_channels=1):
        super(TextDetectionModel, self).__init__()
        
        # Initial double convolution
        self.inc = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64)
        )
        
        # Downsampling path
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 512)
        
        # Upsampling path
        self.up1 = UpBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.up4 = UpBlock(64, 64)
        
        # Output layer - produces segmentation mask
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Enhanced confidence prediction with additional layers
        self.confidence = nn.Sequential(
            ConvBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        
        # Enhanced box regression with additional layers
        self.box_regressor = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, 4, kernel_size=1)
        )
    
    def forward(self, predictions, targets):
        # Text map segmentation loss
        text_map_loss = self.text_map_loss(predictions['text_map'], targets['text_map'])
        
        # Initialize box and confidence losses
        box_loss = torch.tensor(0.0, device=text_map_loss.device)
        confidence_loss = torch.tensor(0.0, device=text_map_loss.device)
        
        batch_size = predictions['text_map'].size(0)
        
        # Debug information
        print(f"targets['boxes'] type: {type(targets['boxes'])}")
        print(f"targets['boxes'] length: {len(targets['boxes'])}")
        
        for i in range(min(batch_size, 2)):  # Just check first couple samples
            if i < len(targets['boxes']):
                print(f"Sample {i} boxes type: {type(targets['boxes'][i])}")
                print(f"Sample {i} boxes length: {len(targets['boxes'][i])}")
                if len(targets['boxes'][i]) > 0:
                    print(f"Sample {i} first box: {targets['boxes'][i][0]}")
        
        # Process each item in batch
        valid_samples = 0
        total_boxes = 0
        
        for i in range(batch_size):
            pred_conf = predictions['confidence'][i]
            pred_boxes = predictions['bbox_coords'][i]
            
            # Skip if no boxes
            if i >= len(targets['boxes']) or not targets['boxes'][i] or len(targets['boxes'][i]) == 0:
                print(f"Sample {i}: No valid boxes found")
                continue
            
            # Get image dimensions
            h, w = pred_conf.shape[1:]
            
            # Process target boxes
            valid_samples += 1
            sample_boxes = targets['boxes'][i]
            total_boxes += len(sample_boxes)
            
            print(f"Processing {len(sample_boxes)} boxes for sample {i}")
            
            for box_idx, box in enumerate(sample_boxes):
                # Extract box data (expected format: [conf, x1, y1, x2, y2])
                if len(box) < 5:
                    print(f"Invalid box format: {box}")
                    continue
                    
                conf, x1, y1, x2, y2 = box
                
                # Normalize coordinates to [0,1]
                x1_norm, y1_norm = x1 / w, y1 / h
                x2_norm, y2_norm = x2 / w, y2 / h
                
                # Sample center point
                center_y = int((y1 + y2) / 2)
                center_x = int((x1 + x2) / 2)
                
                # Check bounds
                if center_y < 0 or center_y >= h or center_x < 0 or center_x >= w:
                    print(f"Center point out of bounds: ({center_x}, {center_y})")
                    continue
                    
                # Get model predictions at center point
                pred_box = pred_boxes[:, center_y, center_x]
                pred_conf = pred_conf[0, center_y, center_x]
                
                # Target box (normalized)
                target_box = torch.tensor([x1_norm, y1_norm, x2_norm, y2_norm], device=pred_box.device)
                target_conf = torch.tensor(conf, device=pred_conf.device)
                
                # Calculate individual box loss
                box_l1_loss = F.smooth_l1_loss(pred_box, target_box)
                box_loss += box_l1_loss
                
                # Calculate confidence loss
                conf_loss = F.binary_cross_entropy(pred_conf, target_conf)
                confidence_loss += conf_loss
                
                print(f"Box {box_idx}: L1 Loss = {box_l1_loss.item()}, Conf Loss = {conf_loss.item()}")
        
        # Average losses
        if valid_samples > 0:
            box_loss /= total_boxes
            confidence_loss /= total_boxes
        
        # Print summary
        print(f"Valid samples: {valid_samples}, Total boxes: {total_boxes}")
        print(f"Box loss: {box_loss.item()}, Confidence loss: {confidence_loss.item()}")
        
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

def get_model(in_channels=3, out_channels=1):
    """Helper function to create the text detection model"""
    model = TextDetectionModel(in_channels, out_channels)
    return model