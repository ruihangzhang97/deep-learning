import torch
from torch import nn
from torch.nn import functional as F

from attention import SelfAttention

class VAE_Attention_Block(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_channels) # each group = in_channels / 32
        self.self_attention = SelfAttention(1, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, height, width)
        residue = x

        n, c, h, w = x.shape

        # (batch_size, features, height, width) -> (batch_size, features, height * width)
        x = x.view(n, c, h * w)

        # (batch_size, features, height * width) -> (batch_size, height * width, features)
        x = x.transpose(-1, -2)

        # (batch_size, height * width, features) -> (batch_size, height * width, features)
        x = self.attention(x)

        # (batch_size, height * width, features) -> (batch_size, features, height * width)
        x = x.transpose(-1, -2)

        # (batch_size, features, height * width) -> (batch_size, features, height, width)
        x = x.view((n, c, h, w))

        x += residue

        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x: (batch_size, in_channels, height, width)
        residue = x

        # x: (batch_size, in_channels, height, width)
        x = self.group_norm1(x)

        # x: (batch_size, in_channels, height, width)
        x = F.silu(x)

        # x: (batch_size, in_channels, height, width)
        x = self.conv1(x)

        # x: (batch_size, in_channels, height, width)
        x = self.group_norm2(x)

        # x: (batch_size, in_channels, height, width)
        x = F.silu(x)

        # x: (batch_size, in_channels, height, width)
        x = self.conv2(x)
    
        return x + self.residual_layer(residue)
    
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            nn.Conv2d(4, 512, kernel_size=3, padding=1), 

            VAE_ResidualBlock(512, 512),

            VAE_AttentionBlock(512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (batch_size, num_features = 512, height/8 = 512/8, width/8 = 512/8) -> (batch_size, num_features = 512, height/8 = 512/8, width/8 = 512/8)
            VAE_ResidualBlock(512, 512),

            # (batch_size, num_features = 512, height/8 = 512/8, width/8 = 512/8) -> (batch_size, num_features = 512, height/4 = 512/4, width/4 = 512/4)
            nn.Upsample(scale_factor=2), 

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            VAE_ResidualBlock(512, 512),

            # (batch_size, num_features = 512, height/4 = 512/4, width/4 = 512/4) -> (batch_size, num_features = 512, height/2 = 512/2, width/2 = 512/2)
            nn.Upsample(scale_factor=2), 

            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 256),

            VAE_ResidualBlock(256, 256),

            VAE_ResidualBlock(256, 256),
            
            # (batch_size, num_features = 256, height/2 = 512/2, width/2 = 512/2) -> (batch_size, num_features = 256, height, width)
            nn.Upsample(scale_factor=2), 

            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            VAE_ResidualBlock(256, 128),

            VAE_ResidualBlock(128, 128),

            VAE_ResidualBlock(128, 128),

            # (32 groups, 128 features): dividing 128 features into groups of 32
            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (batch_size, num_features = 128, height, width) -> (batch_size, num_features = 3, height, width)
            nn.Conv2d(128, kernel_size=3, padding=1)
        )
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 4, height/8, width/8)

        x /= 0.18215

        for module in self:
            x = module(x)

        # (batch_size, 3, height, width)
        return x



        


