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
    
    def forward(self, x):
        """
        Forward pass for the text detection model
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            dict: Dictionary containing 'text_map', 'confidence', and 'bbox_coords'
        """
        # Contracting path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Expanding path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output prediction maps
        text_map = torch.sigmoid(self.outc(x))
        
        # Confidence scores [0,1]
        confidence = torch.sigmoid(self.confidence(x))
        
        # Box coordinates [0,1] for x1,y1,x2,y2
        bbox_coords = torch.sigmoid(self.box_regressor(x))
        
        return {
            'text_map': text_map,          # Text/non-text binary map
            'confidence': confidence,      # Confidence scores
            'bbox_coords': bbox_coords     # Bounding box coordinates (normalized)
        }

def get_model(in_channels=3, out_channels=1):
    """Helper function to create the text detection model"""
    model = TextDetectionModel(in_channels, out_channels)
    return model