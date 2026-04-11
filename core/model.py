import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. LIGHT UNET COMPONENTS
# ==========================================

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UpsampleBlock(nn.Module):
    """
    Bilinear Upsampling + Convolution.
    Smoother than PixelShuffle/TransposeConv, avoids checkerboard artifacts.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class LightUNet(nn.Module):
    def __init__(self, n_channels, n_classes): # n_channels is 64 or 128, n_classes is 4 to get all outputs with 1 model
        super(LightUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Architecture: Light version (32->64->128->256)
        self.inc = DoubleConv(n_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        self.up1 = UpsampleBlock(256, 128)
        self.conv1 = DoubleConv(256, 128)

        self.up2 = UpsampleBlock(128, 64)
        self.conv2 = DoubleConv(128, 64)

        self.up3 = UpsampleBlock(64, 32)
        self.conv3 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x): # -> 256x256x64 (or x128)
        x1 = self.inc(x) # -> 256x256x32
        x2 = self.down1(x1) # -> 128x128x64
        x3 = self.down2(x2) # -> 64x64x128
        x4 = self.down3(x3) # -> 32x32x256

        x = self.up1(x4) # -> 64x64x128
        x = torch.cat([x3, x], dim=1) # -> 64x64x256 # dim1 = channels
        x = self.conv1(x) # -> 64x64x128

        x = self.up2(x) # -> 128x128x64
        x = torch.cat([x2, x], dim=1) # -> 128x128x128
        x = self.conv2(x) # -> 128x128x64

        x = self.up3(x) # -> 256x256x32
        x = torch.cat([x1, x], dim=1) # -> 256x256x64
        x = self.conv3(x) # -> 256x256x32

        logits = self.outc(x) # -> 256x256x4
        return logits


# ==========================================
# 2. SIMPLE DECODER
# ==========================================

class StandardUpsampleBlock(nn.Module):
    """
    Uses standard dense convolutions.
    Blazingly fast on Apple Silicon MPS, unlike grouped/depthwise convs.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        # Standard 3x3 convolution (groups=1) which the M2 GPU loves
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x)
        return self.act(x)


class EfficientDecoder256Fast(nn.Module):
    """
    High-speed, memory-safe decoder for 16x16 -> 256x256 upsampling on M2 Max.
    """

    def __init__(self, in_channels=768, out_channels=4):
        super().__init__()

        # THE SQUEEZE: 768 -> 256 at 16x16 resolution. (Prevents memory blowup)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU()
        )

        # PROGRESSIVE UPSAMPLING: Halving channels as resolution doubles.
        self.up1 = StandardUpsampleBlock(256, 128)  # 16x16   -> 32x32
        self.up2 = StandardUpsampleBlock(128, 64)  # 32x32   -> 64x64
        self.up3 = StandardUpsampleBlock(64, 32)  # 64x64   -> 128x128
        self.up4 = StandardUpsampleBlock(32, 16)  # 128x128 -> 256x256

        # PREDICTION HEAD
        self.head = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):  # 16x16x768
        x = self.bottleneck(x) #  -> 16x16x256
        x = self.up1(x) # -> 32x32x128 
        x = self.up2(x) # -> 64x64x64
        x = self.up3(x) # -> 128x128x32
        x = self.up4(x) # -> 256x256x16
        return self.head(x) # -> 256x256x4



# ==========================================
# 3. SELECT MODEL FUNCTION
# ==========================================


def build_model(model_type, n_channels, n_classes): # n_classes is 4 - to get all outputs with 1 model
    selected = model_type.lower()

    if selected == "auto":
        if n_channels == 768: # provided datasets have then shape 16x16x768
            selected = "decoder_residual"
        else: # provided dataset have shapes 256x256x(64 or 128)
            selected = "lightunet"

    if selected == "lightunet":
        return LightUNet(n_channels, n_classes), selected
    if selected == "decoder_residual": # default in channels 768
        return EfficientDecoder256Fast(in_channels=n_channels, out_channels=n_classes), selected

    raise ValueError(
        f"Unknown model_type '{model_type}'. Use one of: auto, lightunet, decoder_residual"
    )