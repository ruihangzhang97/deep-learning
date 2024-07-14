import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x: (batch_size, channels, height, width)
        residue = x
        batch_size, num_channels, height, width = x.shape
        
        # (batch_size, channels, height, width) -> (batch_size, channels, height * width)
        x = x.view(batch_size, num_channels, height * width)

        # (batch_size, channels, height * width) -> (batch_size, height * width, channels)
        # transposing for attention
        x = x.transpose(-1, -2)

        # (batch_size, height * width, channels) -> (batch_size, height * width, channels)
        x = self.attention(x)

        # (batch_size, height * width, channels) -> (batch_size, channels, height * width)
        x = x.transpose(-1, -2)

        # (batch_size, channels, height * width) -> (batch_size, num_channels, height, width)
        x = x.view(batch_size, num_channels, height, width)

        x += residue
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x: (batch_size, in_channels, height, width)
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue) # making sure x and residue have the same size when in_channels != out_channels