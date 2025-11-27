"""
PyTorch Implementation of 3D Fat Suppression v2 Models

This module provides PyTorch implementations of:
- 3D U-Net with attention mechanisms
- Multi-modal fusion capabilities
- GAN discriminator for adversarial training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np


class ChannelAttention3D(nn.Module):
    """3D Channel Attention Module"""
    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # Ensure at least 1 channel for attention
        reduced_channels = max(1, in_channels // ratio)
        self.fc1 = nn.Conv3d(in_channels, reduced_channels, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(reduced_channels, in_channels, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention3D(nn.Module):
    """3D Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention3D, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_cat)
        return x * self.sigmoid(x_out)


class CBAM3D(nn.Module):
    """3D Convolutional Block Attention Module"""
    def __init__(self, in_channels, ratio=8, kernel_size=7):
        super(CBAM3D, self).__init__()
        self.channel_attention = ChannelAttention3D(in_channels, ratio)
        self.spatial_attention = SpatialAttention3D(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x)
        return x


class ConvBlock3D(nn.Module):
    """3D Convolutional Block with optional attention"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 use_attention=False, ratio=8):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM3D(out_channels, ratio)

        # Residual connection if channels match
        self.residual = (in_channels == out_channels)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_attention:
            out = self.attention(out)

        if self.residual:
            out += residual

        out = self.relu(out)
        return out


class EncoderBlock3D(nn.Module):
    """3D Encoder Block"""
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(EncoderBlock3D, self).__init__()
        self.conv_block = ConvBlock3D(in_channels, out_channels, use_attention=use_attention)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv_block(x)
        skip = x  # Skip connection
        x = self.pool(x)
        return x, skip


class DecoderBlock3D(nn.Module):
    """3D Decoder Block"""
    def __init__(self, in_channels, out_channels, use_attention=True):
        super(DecoderBlock3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, 2, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout3d(0.3)

        self.conv_block = ConvBlock3D(out_channels * 2, out_channels, use_attention=use_attention)

    def forward(self, x, skip):
        x = self.up(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class UNet3D(nn.Module):
    """
    3D U-Net with Attention Mechanisms
    """
    def __init__(self, in_channels=3, out_channels=1, features=16, use_attention=True,
                 num_modalities=1):
        super(UNet3D, self).__init__()

        self.num_modalities = num_modalities
        self.channels_per_modality = in_channels // num_modalities if num_modalities > 1 else in_channels

        # Input processing
        if num_modalities > 1:
            self.modality_attention = nn.ModuleList([
                CBAM3D(self.channels_per_modality) for _ in range(num_modalities)
            ])
        else:
            self.input_attention = CBAM3D(in_channels)

        # Encoder
        self.enc1 = EncoderBlock3D(in_channels, features, use_attention)
        self.enc2 = EncoderBlock3D(features, features*2, use_attention)
        self.enc3 = EncoderBlock3D(features*2, features*4, use_attention)
        self.enc4 = EncoderBlock3D(features*4, features*8, use_attention)

        # Bottleneck
        self.bottleneck = ConvBlock3D(features*8, features*16, use_attention=use_attention)

        # Decoder
        self.dec4 = DecoderBlock3D(features*16, features*8, use_attention)
        self.dec3 = DecoderBlock3D(features*8, features*4, use_attention)
        self.dec2 = DecoderBlock3D(features*4, features*2, use_attention)
        self.dec1 = DecoderBlock3D(features*2, features, use_attention)

        # Output
        self.final_conv = nn.Conv3d(features, out_channels, 1)

    def forward(self, x):
        # Multi-modal fusion
        if self.num_modalities > 1:
            # Split modalities
            modalities = []
            for i in range(self.num_modalities):
                start_ch = i * self.channels_per_modality
                end_ch = (i + 1) * self.channels_per_modality
                modality = x[:, start_ch:end_ch, :, :, :]

                # Apply modality-specific attention
                modality = self.modality_attention[i](modality)
                modalities.append(modality)

            # Fuse modalities (concatenate)
            x = torch.cat(modalities, dim=1)
        else:
            # Single modality attention
            x = self.input_attention(x)

        # Encoder
        enc1, skip1 = self.enc1(x)
        enc2, skip2 = self.enc2(enc1)
        enc3, skip3 = self.enc3(enc2)
        enc4, skip4 = self.enc4(enc3)

        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        dec4 = self.dec4(bottleneck, skip4)
        dec3 = self.dec3(dec4, skip3)
        dec2 = self.dec2(dec3, skip2)
        dec1 = self.dec1(dec2, skip1)

        # Output
        out = self.final_conv(dec1)
        out = torch.sigmoid(out)

        return out


class Discriminator3D(nn.Module):
    """3D Discriminator for GAN training"""
    def __init__(self, in_channels=1, features=64):
        super(Discriminator3D, self).__init__()

        # Use smaller kernel sizes and strides to handle smaller inputs
        self.model = nn.Sequential(
            # Input: (batch, channels, depth, height, width)
            nn.Conv3d(in_channels, features, 3, 2, 1, bias=False),  # Changed from 4 to 3
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(features, features*2, 3, 2, 1, bias=False),  # Changed from 4 to 3
            nn.BatchNorm3d(features*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(features*2, features*4, 3, 2, 1, bias=False),  # Changed from 4 to 3
            nn.BatchNorm3d(features*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(features*4, features*8, 3, 2, 1, bias=False),  # Changed from 4 to 3
            nn.BatchNorm3d(features*8),
            nn.LeakyReLU(0.2, inplace=True),

            # Adaptive pooling to handle variable input sizes
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(features*8, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def create_3d_generator(in_channels=3, features=16, num_modalities=1, use_attention=True):
    """Create 3D U-Net generator"""
    model = UNet3D(in_channels, 1, features, use_attention, num_modalities)
    init_weights(model)
    return model


def create_3d_discriminator(in_channels=1, features=64):
    """Create 3D discriminator"""
    model = Discriminator3D(in_channels, features)
    init_weights(model)
    return model


# Test the models
if __name__ == "__main__":
    # Test single modality
    print("Testing 3D U-Net models...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Test generator
    generator = create_3d_generator(in_channels=3, num_modalities=1)
    generator.to(device)

    # Test input
    test_input = torch.randn(1, 3, 32, 128, 128).to(device)
    with torch.no_grad():
        output = generator(test_input)
    print(f"Generator: {test_input.shape} -> {output.shape}")

    # Test discriminator
    discriminator = create_3d_discriminator()
    discriminator.to(device)

    disc_input = torch.randn(1, 1, 32, 128, 128).to(device)
    with torch.no_grad():
        disc_output = discriminator(disc_input)
    print(f"Discriminator: {disc_input.shape} -> {disc_output.shape}")

    # Test multi-modal
    multi_generator = create_3d_generator(in_channels=6, num_modalities=2)
    multi_generator.to(device)

    multi_input = torch.randn(1, 6, 32, 128, 128).to(device)
    with torch.no_grad():
        multi_output = multi_generator(multi_input)
    print(f"Multi-modal Generator: {multi_input.shape} -> {multi_output.shape}")

    print("✅ All 3D models working correctly!")
