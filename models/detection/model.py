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
        
        # Important: After concatenation, we have in_channels // 2 + out_channels
        # channels, not in_channels // 2 + in_channels as you might have written
        self.conv1 = ConvBlock(in_channels // 2 + out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Adjust dimensions if needed
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, 
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class AttentionBlock(nn.Module):
    """Attention gate for focused feature selection"""
    def __init__(self, f_g, f_l, f_int=None):
        super(AttentionBlock, self).__init__()
        if f_int is None:
            f_int = min(f_g, f_l) // 2  # Use half the minimum of both channels
        
        # Gating signal projection
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        
        # Feature map projection
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        
        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Contracting path
        x1 = self.inc(x)  # Output: 64 channels
        x2 = self.down1(x1)  # Output: 128 channels
        x3 = self.down2(x2)  # Output: 256 channels
        x4 = self.down3(x3)  # Output: 512 channels
        x5 = self.down4(x4)  # Output: 1024 channels
        
        # IMPORTANT FIX: The parameters order in AttentionBlock.forward is (g, x)
        # where g is the gating signal (lower resolution feature map) and
        # x is the feature map being attended to (skip connection)
        # So it should be self.att1(x5, x4) not self.att1(x4, x5)
        x4_att = self.att1(x5, x4)  # Correct: gating signal x5, feature map x4
        
        # Here x5 has 1024 channels, x4_att has 512 channels
        x = self.up1(x5, x4_att)  # Output: 512 channels
        
        # Fix the same issue in the remaining attention blocks
        x3_att = self.att2(x, x3)  # Correct: gating signal x, feature map x3
        x = self.up2(x, x3_att)  # Output: 256 channels
        
        x2_att = self.att3(x, x2)  # Correct: gating signal x, feature map x2
        x = self.up3(x, x2_att)  # Output: 128 channels
        
        x1_att = self.att4(x, x1)  # Correct: gating signal x, feature map x1
        x = self.up4(x, x1_att)  # Output: 64 channels
        
        # Output prediction maps
        text_map = torch.sigmoid(self.outc(x))
        confidence = torch.sigmoid(self.confidence(x))
        bbox_coords = torch.sigmoid(self.box_regressor(x))
        
        return {
            'text_map': text_map,
            'confidence': confidence,
            'bbox_coords': bbox_coords
        }
        
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
        self.down4 = DownBlock(512, 1024)
        
        # Upsampling path
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        
        # Attention gates for better feature refinement
        self.att1 = AttentionBlock(512, 512)
        self.att2 = AttentionBlock(256, 256)
        self.att3 = AttentionBlock(128, 128)
        self.att4 = AttentionBlock(64, 64)
        
        # Output layers
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Separate branch for confidence prediction
        self.confidence = nn.Sequential(
            ConvBlock(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        
        # Box regression with multi-scale feature fusion
        self.box_regressor = nn.Sequential(
            ConvBlock(64, 64),
            ConvBlock(64, 32),
            nn.Conv2d(32, 4, kernel_size=1)
        )
    
    def forward(self, x):
        # Contracting path
        x1 = self.inc(x)  # Output: 64 channels
        x2 = self.down1(x1)  # Output: 128 channels
        x3 = self.down2(x2)  # Output: 256 channels
        x4 = self.down3(x3)  # Output: 512 channels
        x5 = self.down4(x4)  # Output: 1024 channels
        
        # Apply attention mechanism to skip connections
        # Make sure x4_att has the same number of channels as x4 (512)
        x4_att = self.att1(x4, x5)  # Important: This should output 512 channels
        
        # Here x5 has 1024 channels, x4_att has 512 channels
        x = self.up1(x5, x4_att)  # Output: 512 channels
        
        # Continue similarly with the rest of the upsampling path
        # Making sure channel dimensions match at each step
        x3_att = self.att2(x3, x)  # Input: x3 (256 channels), x (512 channels)
        x = self.up2(x, x3_att)  # Output: 256 channels
        
        x2_att = self.att3(x2, x)  # Input: x2 (128 channels), x (256 channels)
        x = self.up3(x, x2_att)  # Output: 128 channels
        
        x1_att = self.att4(x1, x)  # Input: x1 (64 channels), x (128 channels)
        x = self.up4(x, x1_att)  # Output: 64 channels
        
        # Output prediction maps
        text_map = torch.sigmoid(self.outc(x))  # Input: 64 channels
        confidence = torch.sigmoid(self.confidence(x))  # Input: 64 channels
        bbox_coords = torch.sigmoid(self.box_regressor(x))  # Input: 64 channels
        
        return {
            'text_map': text_map,
            'confidence': confidence,
            'bbox_coords': bbox_coords
        }

def get_model(in_channels=3, out_channels=1):
    """Helper function to create the text detection model"""
    model = TextDetectionModel(in_channels, out_channels)
    return model