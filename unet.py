import torch
from torch import nn
from torch.nn import functional as F


class SwitchSequential(nn.Sequential):

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                 time: torch.Tensor) -> torch.Tensor:

        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        
        return x

class UNET(nn.Module):

    def __init__(self, u_dim: int=320, in_channels:int = 4, 
                 n_head: int=8, n_emdb=40):
        super().__init__()    
    
        self.encoders = nn.ModuleList([
            SwitchSequential(nn.Conv2d(in_channels, u_dim, 
            # (batch_size, 4, h/8, w/8)
                                       kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(u_dim, u_dim), 
                             UNET_AttentionBlock(n_head, n_emdb)),
            SwitchSequential(UNET_ResidualBlock(u_dim, u_dim), 
                             UNET_AttentionBlock(n_head, n_emdb)),

            # (batch_size, u_dim, h/8, w/8) -> (batch_size, u_dim, h/16, w/16) 
            SwitchSequential(nn.Conv2d(u_dim, u_dim, kernel_size=3, 
                                        stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(u_dim, 2 * u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb)),
            SwitchSequential(UNET_ResidualBlock(2 * u_dim, 2 * u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb)),

            # (batch_size, 2 * u_dim, h/16, w/16) -> (batch_size, 2 * u_dim, h/32, w/32) 
            SwitchSequential(nn.Conv2d(2 * u_dim, 2 * u_dim,kernel_size=3, 
                                       stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(2 * u_dim, 4 * u_dim), 
                             UNET_AttentionBlock(n_head, 4 * n_emdb)),
            SwitchSequential(UNET_ResidualBlock(4 * u_dim, 4 * u_dim), 
                             UNET_AttentionBlock(n_head, 4 * n_emdb)),

            # (batch_size, 4 * u_dim, h/32, w/32) -> (batch_size, 4 * u_dim, h/64, w/64) 
            SwitchSequential(nn.Conv2d(4 * u_dim, 2 * u_dim,kernel_size=3, 
                                       stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(4 * u_dim, 4 * u_dim)),

            # (batch_size, 4 * u_dim, h/64, w/64) -> same
            SwitchSequential(UNET_ResidualBlock(4 * u_dim, 4 * u_dim))

        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(4 * u_dim, 4 * u_dim), 
            UNET_AttentionBlock(h_head, 4 * n_emdb),
            UNET_ResidualBlock(4 * u_dim, 4 * u_dim))
        
        self.decoders = nn.ModuleList([

            # (batch_size, 8 * u_dim, h/64, w/64) -> (batch_size, 4 * u_dim, h/64, w/64)
            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim)),

            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim)),

            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim), 
                             UpSample(4 * u_dim)),

            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim), 
                             UNET_AttentionBlock(n_head, 4 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim), 
                             UNET_AttentionBlock(n_head, 4 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(6 * u_dim, 4 * u_dim), 
                             UNET_AttentionBlock(n_head, 4 * n_emdb),
                             UpSample(4 * u_dim)),

            
            SwitchSequential(UNET_ResidualBlock(6 * u_dim, 2 * u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(4 * u_dim, 2 * u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(3 * u_dim, 2 * u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb),
                             UpSample(2 * u_dim)),

            SwitchSequential(UNET_ResidualBlock(3 * u_dim, u_dim), 
                             UNET_AttentionBlock(n_head, n_emdb)),

            SwitchSequential(UNET_ResidualBlock(2 * u_dim, u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(2 * u_dim, u_dim), 
                             UNET_AttentionBlock(n_head, n_emdb)),

        ])


#UNET_OutputLayer
