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


class VAE_Decoder(nn.Sequential):

    def __init__(self, input_channels=3):

        super().__init__(

            # handle bottleneck output
            # (batch_size, 4, h/8, w/8) -> same
            nn.Conv2d(4, 4, kernel_size=1, padding=0),

            # (batch_size, 4, h/8, w/8) -> (batch_size, 512, h/8, w/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            # (batch_size, 512, h/8, w/8) -> same
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/8, w/8) -> same
            VAE_AttentionBlock(512),

            # 4X (batch_size, 512, h/8, w/8) -> same
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/8, w/8) -> (batch_size, 512, h/4, w/4)
            nn.Upsample(scale_factor=2),

            # (batch_size, 512, h/4, w/4) -> same
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # 3 X (batch_size, 512, h/4, w/4) -> same
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (batch_size, 512, h/4, w/4) -> (batch_size, 512, h/2, w/2)
            nn.Upsample(scale_factor=2),

            # (batch_size, 512, h/2, w/2) -> same
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            # (batch_size, 512, h/2, w/2) -> (batch_size, 256, h/2, w/2) 
            VAE_ResidualBlock(512, 256),

            # 2X (batch_size, 256, h/2, w/2) -> same
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (batch_size, 256, h/2, w/2) -> (batch_size, 256, h, w)
            nn.Upsample(scale_factor=2),

            # (batch_size, 256, h, w) -> same
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # (batch_size, 256, h, w) -> (batch_size, 128, h, w) 
            VAE_ResidualBlock(256, 128),

            # 2X (batch_size, 128, h, w) -> same
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
 
            nn.GroupNorm(32, 128),

            nn.SiLU(),

            # (batch_size, 128, h, w) -> Input samples shape :(batch_size, 3, h, w)
            nn.Conv2d(128, input_channels, kernel_size=3, padding=1),

        )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x: (Batch_size, 4, h/8, w/8)

        x /= 0.18215

        for module in self:
            x = module(x)

        #x: (Batch_size, input_channels, h, w)
        return(x)
    

if __name__ == "__main__":
    
    h_w = 512
    # x = torch.Tensor(4, 3, h_w, h_w)
    latent =  torch.Tensor(4, 4, int(h_w/8), int(h_w/8))
    print(latent.shape)    
    model = VAE_Decoder()
    y = model(latent)
    print(y.shape)