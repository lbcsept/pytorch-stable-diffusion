import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention




class VAE_AttentionBlock(nn.Module):
    
    def __init__(self, channels:int, groupnorm:int=32):
        
        super().__init__()

        # normalize by group of 32 layers (locally nearby, 
        # so supposed to have naturally close distribution)
        self.groupnorm = nn.GroupNorm(groupnorm, channels)
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

    def __init__(self, in_channels:int, out_channels:int, groupnorm:int=32):
        
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(groupnorm, in_channels)
        self.conv_1= nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(groupnorm, out_channels)
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

    def __init__(self, input_channels=3, vae_dim:int=512, latent_dim:int=8, groupnorm:int=32, latent_const=0.18215):
        
        self.latent_const=latent_const

        super().__init__(

            # handle bottleneck output
            # (batch_size, 4, h/8, w/8) -> same
            nn.Conv2d(int(latent_dim/2), int(latent_dim/2), kernel_size=1, padding=0),

            # (batch_size, 4, h/8, w/8) -> (batch_size, vae_dim, h/8, w/8)
            nn.Conv2d(int(latent_dim/2), vae_dim, kernel_size=3, padding=1),

            # (batch_size, vae_dim, h/8, w/8) -> same
            VAE_ResidualBlock(vae_dim, vae_dim),

            # (batch_size, vae_dim, h/8, w/8) -> same
            VAE_AttentionBlock(vae_dim),

            # 4X (batch_size, vae_dim, h/8, w/8) -> same
            VAE_ResidualBlock(vae_dim, vae_dim),
            VAE_ResidualBlock(vae_dim, vae_dim),
            VAE_ResidualBlock(vae_dim, vae_dim),
            VAE_ResidualBlock(vae_dim, vae_dim),

            # (batch_size, vae_dim, h/8, w/8) -> (batch_size, vae_dim, h/4, w/4)
            nn.Upsample(scale_factor=2),

            # (batch_size, vae_dim, h/4, w/4) -> same
            nn.Conv2d(vae_dim, vae_dim, kernel_size=3, padding=1),

            # 3 X (batch_size, vae_dim, h/4, w/4) -> same
            VAE_ResidualBlock(vae_dim, vae_dim),
            VAE_ResidualBlock(vae_dim, vae_dim),
            VAE_ResidualBlock(vae_dim, vae_dim),

            # (batch_size, vae_dim, h/4, w/4) -> (batch_size, vae_dim, h/2, w/2)
            nn.Upsample(scale_factor=2),

            # (batch_size, vae_dim, h/2, w/2) -> same
            nn.Conv2d(vae_dim, vae_dim, kernel_size=3, padding=1),

            # (batch_size, vae_dim, h/2, w/2) -> (batch_size, int(vae_dim/2), h/2, w/2) 
            VAE_ResidualBlock(vae_dim, int(vae_dim/2)),

            # 2X (batch_size, int(vae_dim/2), h/2, w/2) -> same
            VAE_ResidualBlock(int(vae_dim/2), int(vae_dim/2)),
            VAE_ResidualBlock(int(vae_dim/2), int(vae_dim/2)),

            # (batch_size, int(vae_dim/2), h/2, w/2) -> (batch_size, int(vae_dim/2), h, w)
            nn.Upsample(scale_factor=2),

            # (batch_size, int(vae_dim/2), h, w) -> same
            nn.Conv2d(int(vae_dim/2), int(vae_dim/2), kernel_size=3, padding=1),

            # (batch_size, int(vae_dim/2), h, w) -> (batch_size, int(vae_dim/4), h, w) 
            VAE_ResidualBlock(int(vae_dim/2), int(vae_dim/4)),

            # 2X (batch_size, int(vae_dim/4), h, w) -> same
            VAE_ResidualBlock(int(vae_dim/4), int(vae_dim/4)),
            VAE_ResidualBlock(int(vae_dim/4), int(vae_dim/4)),
 
            nn.GroupNorm(groupnorm, int(vae_dim/4)),

            nn.SiLU(),

            # (batch_size, int(vae_dim/4), h, w) -> Input samples shape :(batch_size, 3, h, w)
            nn.Conv2d(int(vae_dim/4), input_channels, kernel_size=3, padding=1),

        )


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        #x: (Batch_size, 4, h/8, w/8)

        x /= self.latent_const

        for module in self:
            x = module(x)

        #x: (Batch_size, input_channels, h, w)
        return(x)
    

if __name__ == "__main__":
    
    h_w = 512
    batch_size = 4
    latent_dim = 8
    input_channels = 4
    vae_dim = 512 #512

    # x = torch.Tensor(4, 3, h_w, h_w)
    latent =  torch.Tensor(batch_size, int(latent_dim/2), int(h_w/8), int(h_w/8))
    print(latent.shape)    
    model = VAE_Decoder(input_channels=input_channels, 
                        vae_dim=vae_dim, latent_dim=latent_dim)
    y = model(latent)
    print(y.shape)



