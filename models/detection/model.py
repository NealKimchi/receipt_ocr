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

class AttentionBlock(nn.Module):
    """Attention gate for focused feature selection"""
    def __init__(self, f_g, f_l, f_int=None):
        super(AttentionBlock, self).__init__()
        if f_int is None:
            f_int = f_l
        
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
    
    def forward(self, g, x):
        # Adjust spatial dimensions if needed
        if g.size()[2:] != x.size()[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)
            
        # Apply convolutions
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Element-wise sum and ReLU
        psi = self.relu(g1 + x1)
        
        # Attention map
        psi = self.psi(psi)
        
        # Apply attention
        return x * psi
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
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Apply attention mechanism to skip connections
        x4_att = self.att1(x4, x5)
        x = self.up1(x5, x4_att)
        
        x3_att = self.att2(x3, x)
        x = self.up2(x, x3_att)
        
        x2_att = self.att3(x2, x)
        x = self.up3(x, x2_att)
        
        x1_att = self.att4(x1, x)
        x = self.up4(x, x1_att)
        
        # Output prediction maps
        text_map = torch.sigmoid(self.outc(x))
        confidence = torch.sigmoid(self.confidence(x))
        bbox_coords = torch.sigmoid(self.box_regressor(x))
        
        return {
            'text_map': text_map,
            'confidence': confidence,
            'bbox_coords': bbox_coords
        }

def get_model(in_channels=3, out_channels=1):
    """Helper function to create the text detection model"""
    model = TextDetectionModel(in_channels, out_channels)
    return model