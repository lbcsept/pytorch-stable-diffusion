import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    

    def __init__(self):

        super().__init__(

            # (Batch_size, Channel, Height, Width) -> (Batch_size, 128, Height, Width)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            # (Batch_size, 128, Height, Width)-> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, Height, Width)-> (Batch_size, 128, Height, Width)
            VAE_ResidualBlock(128, 128),

            # (Batch_size, 128, Height, Width) -> (Batch_size, 128, Height/2, Width/2)
            # £assym
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            # (Batch_size, 128, Height/2, Width/2)-> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(128, 256),

            # (Batch_size, 256, Height/2, Width/2)-> (Batch_size, 256, Height/2, Width/2)
            VAE_ResidualBlock(256, 256),

            # (Batch_size, 256, Height/2, Width/2) -> (Batch_size, 512, Height/4, Width/4)
            # £assym
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

            # (Batch_size, 256, Height/2, Width/2)-> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(256, 512),

            # (Batch_size, 512, Height/4, Width/4)-> (Batch_size, 512, Height/4, Width/4)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/4, Width/4) -> (Batch_size, 1024, Height/8, Width/8)
            # £assym
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

            # (Batch_size, 512, Height/8, Width/8)-> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8)-> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8)-> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # (Batch_size, 512, Height/8, Width/8)-> (Batch_size, 512, Height/8, Width/8)
            VAE_AttentionBlock(512),

            # (Batch_size, 512, Height/8, Width/8)-> (Batch_size, 512, Height/8, Width/8)
            VAE_ResidualBlock(512, 512),

            # Normalization (32 groups, 512 features)
            nn.GroupNorm(32, 512),

            # activation : SILU (could also be RELU, but pratically SILU seems better)
            nn.SiLU(),

            # Bottleneck
            # (Batch_size, 512, Height/8, Width/8) -> (Batch_size, 512, Height/8, Width/8) 
            nn.Conv2d(512, 8, kernel_size=3, padding=1),

            # (Batch_size, 8, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8) re
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch_Size, Channel, Height, Width)
        noise: (Batch_Size, Output_Channels, Height/8, Width/8)
        """

        # applying an assetric padding on layers tagged # £assym above.
        # not clear why we do that, but it seems to be implemented this way in 
        # AVE models re-used by stable diffusion 
        # if we don't do this torch.Size([4, 3, 512, 512]) -> torch.Size([4, 8, 63, 63])
        # if we don't do this torch.Size([4, 3, 512, 512]) -> torch.Size([4, 8, 64, 64])        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # (padding left, Right, Top, Bot),
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        return x

if __name__ == "__main__":
    
    x = torch.Tensor(4, 3, 1024, 1024)
    print(x.shape)    
    model = VAE_Encoder()
    y = model(x, x)
    print(y.shape)