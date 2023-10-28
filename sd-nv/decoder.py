import torch
from torch import nn




class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, in_channel):
        super().__init__()
        self.conv= nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)


    def forward(self, x):
        return self.conv(x)


class VAE_ResidualBlock(nn.Module):

    def __init__(self, in_channel, out_channel):
        super().__init__()
        # (Batch_size, Channel, Height, Width) -> (Batch_size, 128, Height, Width)
        self.conv= nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)


    def forward(self, x):
        return self.conv(x)
