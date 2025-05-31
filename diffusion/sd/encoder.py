import torch
from torch import nn
from torch.nn import functional as F

from decoder import VAE_Attention_Block, VAE_ResidualBlock

class VAE_Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256, num_layers=4, num_filters=64):
        super().__init__(
            # (Batch_size, Channels, Height, Width) -> (Bathch_size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),  # residual block doesnt change image dimensions

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128), 

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height/2, Width/2)
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0), 

            # (Batch_size, 128, Height/2, Width/2) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256), 

            # (Batch_size, 256, Height/2, Width/2) -> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256), 

            # (Batch_size, 256, Height/2, Width/2) -> (Batch_size, 256, Height/4, Width/4)
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 

            # (Batch_size, 256, Height/2, Width/2) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512), 

            # (Batch_size, 512, Height/4, Width/4) -> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/4, Width/4) -> (Batch_size, 512, Height/8, Width/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_Attention_Block(512), 

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512), 

            nn.SiLU(),

            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),    # bottle neck of encoder

            # (Batch_size, 8, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch_size, Channels, Height, Width)
        # noise: (Batch_size, Channels, Height, Width)

        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (Padding_left, Padding_right, Padding_top, Padding_bottom)
                x = F.pad(x, (0, 1, 0, 1))
            
            # compressed version of the image x
            x = module(x)

        # x: (Batch_size, 8, Height/8, Width/8) -> mean: (Batch_size, 4, Height/8, Width/8), log_variance: (Batch_size, 4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)  # splits into mean and log_variance

        # log_variance: (Batch_size, 4, Height/8, Width/8)
        log_variance = torch.clamp(log_variance, -30.0, 20.0)  # clamp log_variance to avoid numerical issues

        # variance: (Batch_size, 4, Height/8, Width/8)
        variance = log_variance.exp()  # variance = exp(log_variance)

        # stdev: (Batch_size, 4, Height/8, Width/8)
        stdev = variance.sqrt()

        # Z = N(0, 1) -> N(mean, variance)?
        # reparameterization trick to sample from the distribution N(mean, variance)
        # X = mean + stdev * Z
        x = mean + stdev * noise  # noise is sampled from N(0, 1)

        # scale the output by a constant factor
        # x: (Batch_size, 4, Height/8, Width/8)
        x = x * 0.18215
        
        return x, mean, log_variance


        






