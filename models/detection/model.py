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
        # Transpose convolution reduces channels by half
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        # After concatenation with skip connection, we'll have (in_channels // 2 + out_channels) channels
        self.conv1 = ConvBlock(in_channels // 2 + out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x1, x2):
        # x1 is from the deeper layer (e.g., from down path)
        # x2 is the skip connection from the encoder
        x1 = self.up(x1)  # This halves the number of channels in x1
        
        # Adjust dimensions if needed (for height and width)
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
    
    def forward(self, g, x):
        """
        Attention mechanism
        Args:
            g: Gating signal (B, C, H, W)
            x: Skip connection feature map (B, C, H, W)
        """
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
        
        # Apply attention - this does not change the number of channels in x
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
        
        # Attention gates - each attention gate has parameters for gating signal channels and skip connection channels
        # The gating signal comes from the decoder path, and the skip connection from the encoder
        self.att1 = AttentionBlock(f_g=1024, f_l=512)  # f_g: gating signal channels, f_l: skip connection channels
        self.att2 = AttentionBlock(f_g=512, f_l=256)
        self.att3 = AttentionBlock(f_g=256, f_l=128)
        self.att4 = AttentionBlock(f_g=128, f_l=64)
        
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
        x1 = self.inc(x)           # Output: 64 channels
        x2 = self.down1(x1)        # Output: 128 channels
        x3 = self.down2(x2)        # Output: 256 channels
        x4 = self.down3(x3)        # Output: 512 channels
        x5 = self.down4(x4)        # Output: 1024 channels
        
        # Apply attention mechanism to skip connections
        # Important: Attention is applied to skip connections BEFORE they're used in the upsampling path
        x4_att = self.att1(x5, x4)  # Gating signal: x5 (1024), Skip connection: x4 (512) -> Output: x4_att (512 channels)
        
        # Up path with attended features
        x = self.up1(x5, x4_att)   # Input: x5 (1024), x4_att (512) -> Output: 512 channels
        
        x3_att = self.att2(x, x3)  # Gating signal: x (512), Skip connection: x3 (256) -> Output: x3_att (256 channels)
        x = self.up2(x, x3_att)    # Input: x (512), x3_att (256) -> Output: 256 channels
        
        x2_att = self.att3(x, x2)  # Gating signal: x (256), Skip connection: x2 (128) -> Output: x2_att (128 channels)
        x = self.up3(x, x2_att)    # Input: x (256), x2_att (128) -> Output: 128 channels
        
        x1_att = self.att4(x, x1)  # Gating signal: x (128), Skip connection: x1 (64) -> Output: x1_att (64 channels)
        x = self.up4(x, x1_att)    # Input: x (128), x1_att (64) -> Output: 64 channels
        
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