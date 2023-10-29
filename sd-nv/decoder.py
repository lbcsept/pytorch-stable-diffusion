import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention




class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, channels):
        super().__init__()

        # normalize by group of 32 layers (locally nearby, 
        # so supposed to have naturally close distribution)
        self.groupnorm = nn.GroupNorm(32, channels)
        # self attention between all pixels in image
        self.attention = SelfAttention(n_heads=1, d_embed=channels)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, height, width)
        residue = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)

        n, c, h, w = x.shape

        # -> (batch_size, features, height * width) 
        x = x.view(n, c, h * w)
        
        # -> (batch_size, height * width, features) 
        x = x.transpose(-1, -2)

        # Perform self-attention WITHOUT mask
        # keep shape
        x = self.attention(x)

        # -> (batch_size, fatures, height * width) 
        x = x.transpose(-1, -2)
    
        # -> (batch_size, features, height, width) 
        x = x.view(n, c, h, w)

        return x + residue




class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1= nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2= nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, 
                                                         kernel_size=1, padding=0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # x: (Batch_Size, in_channels, height, width)

        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x)

        x = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residual_layer(residue)


