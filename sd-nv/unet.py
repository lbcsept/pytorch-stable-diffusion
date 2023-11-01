import torch
from torch import nn
from torch.nn import functional as F
from decoder import SelfAttention

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
    

class UNET_ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels:int, n_time: int=4 * 380,
                 groupnorm=32):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(groupnorm, in_channels)
        self.conv_feature = nn.conv2d(in_channels, out_channels, kernel_size=3, padding=1) 
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(groupnorm, out_channels)
        self.conv_merged = nn.conv2d(out_channels, out_channels, kernel_size=3, padding=1) 


        if in_channels == out_channels:
            self.residual_layer = nn.Identity() 
        else:
            self.residual_layer = nn.conv2d(in_channels, out_channels, kernel_size=1, padding=0) 
    
    def forward(self, feature, time):
        # (batch_size, in_channels, h, w)
        # time(1, 1280)

        residue = feature

        feature = self.groupnorm_feature(feature)

        feature = F.silu(feature)
        
        feature = self.conv_feature(feature)

        time = F.silu(time)

        time = self.linear_time(time)

        # adding dimension to time because it doesn't have batch and inchannel
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)

        merged =  self.groupnorm_merged(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)
    


class UNET_AttentionBlock(nn.Module):

    def __init__(self, n_head: int, n_embd: int, d_context: int=768, groupnorm: int=32):
        super().__init__()
        channels = n_head * n_embd

        self.groupnorm = nn.GroupNorm(groupnorm,  channels, eps =1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.conv2d(channels, channels, kernel_size=1, padding=0)


    def forward(self, x, context):
        # x: (batch_size, features, h, w)
        # context: (batch_size, seq_len, dim)

        residue_long = x 

        x = self.groupnorm(x)
        
        x = self.conv_input(x)

        n, c, h, w = x.shape

        # (batch_size, features, h, w) -> (batch_size, features, h * w)
        x = x.view((n, c, h*w))

        # (batch_size, features, h * w) -> (batch_size, h * w, features)
        x =  x.transpose(-1, -2) 

        # normalization + self attention with skip connection

        residue_short = x

        x =  self.layernorm_1(x)
        
        x =  self.attention_1(x)

        x += residue_short

        residue_short = x

        # normalization + cross attention with skip connection
        x = self.layernorm_2(x)

        # Cross Attention
        x = self.attention_2(x, context)

        x += residue_short

        residue_short = x

        # Normalization + FF with GeGlu and skip connection 

        x = self.layernorm_3(x)
        
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_geglu_2(x)

        x *= residue_short

        # (batch_size, h * w, features) -> (batch_size, features, h * w) 
        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        return self.conv_output(x) + residue_long


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
            UNET_AttentionBlock(n_head, 4 * n_emdb),
            UNET_ResidualBlock(4 * u_dim, 4 * u_dim))
        
        self.decoders = nn.ModuleList([

            # (batch_size, 8 * u_dim, h/64, w/64) -> (batch_size, 4 * u_dim, h/64, w/64)
            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim)),

            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim)),

            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim), 
                             Upsample(4 * u_dim)),

            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim), 
                             UNET_AttentionBlock(n_head, 4 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(8 * u_dim, 4 * u_dim), 
                             UNET_AttentionBlock(n_head, 4 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(6 * u_dim, 4 * u_dim), 
                             UNET_AttentionBlock(n_head, 4 * n_emdb),
                             Upsample(4 * u_dim)),

            
            SwitchSequential(UNET_ResidualBlock(6 * u_dim, 2 * u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(4 * u_dim, 2 * u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(3 * u_dim, 2 * u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb),
                             Upsample(2 * u_dim)),

            SwitchSequential(UNET_ResidualBlock(3 * u_dim, u_dim), 
                             UNET_AttentionBlock(n_head, n_emdb)),

            SwitchSequential(UNET_ResidualBlock(2 * u_dim, u_dim), 
                             UNET_AttentionBlock(n_head, 2 * n_emdb)),

            SwitchSequential(UNET_ResidualBlock(2 * u_dim, u_dim), 
                             UNET_AttentionBlock(n_head, n_emdb)),

        ])


class Upsample(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        # (batch_size, features, h, w) -> # (batch_size, features, 2 * h, 2 * w) 
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        return self.conf(x)


class UNET_OutputLayer(nn.Module): 

    def __init__(self, in_channels: int, out_channels: int, 
                 groupnorm=32, time_dim: int=320):
        super().__init__()
        # self.latent_dim = 2 * out_channelq
        self.groupnorm = nn.GroupNorm(groupnorm, in_channels)
        self.conv = nn.conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # (batch, time_dim, h/8, w/8) -> (batch, latent_dim/2, h/8, w/8) 

        x = self.groupnorm(x)

        x = F.silu(x)

        x = self.conv(x)

        #(batch, latent_dim/2, h/8, w/8) 
        return x