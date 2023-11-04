import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Sequential):
    def __init__(
        self,
        input_channels: int = 3,
        vae_dim: int = 512,
        latent_dim: int = 8,
        groupnorm: int = 32,
        latent_const: float = 0.18215,
    ):
        self.input_channels = input_channels
        self.latent_const = latent_const
        self.vae_dim = vae_dim
        self.latent_dim = latent_dim

        super().__init__(
            # (Batch_size, Channel, Height, Width) -> (Batch_size, int(vae_dim/4), Height, Width)
            nn.Conv2d(input_channels, int(vae_dim / 4), kernel_size=3, padding=1),
            # 2X (Batch_size, int(vae_dim/4), Height, Width) -> same
            VAE_ResidualBlock(int(vae_dim / 4), int(vae_dim / 4)),
            VAE_ResidualBlock(int(vae_dim / 4), int(vae_dim / 4)),
            # £assym(Batch_size, int(vae_dim/4), Height, Width) -> (Batch_size, int(vae_dim/4), Height/2, Width/2)
            nn.Conv2d(
                int(vae_dim / 4), int(vae_dim / 4), kernel_size=3, stride=2, padding=0
            ),
            # (Batch_size, int(vae_dim/4), Height/2, Width/2) -> (Batch_size, int(vae_dim/2), Height/2, Width/2)
            VAE_ResidualBlock(int(vae_dim / 4), int(vae_dim / 2)),
            # (Batch_size, int(vae_dim/2), Height/2, Width/2) -> same
            VAE_ResidualBlock(int(vae_dim / 2), int(vae_dim / 2)),
            # £assym (Batch_size, int(vae_dim/2), Height/2, Width/2) -> (Batch_size, vae_dim, Height/4, Width/4)
            nn.Conv2d(
                int(vae_dim / 2), int(vae_dim / 2), kernel_size=3, stride=2, padding=0
            ),
            # (Batch_size, int(vae_dim/2), Height/2, Width/2) -> (Batch_size, vae_dim, Height/4, Width/4)
            VAE_ResidualBlock(int(vae_dim / 2), vae_dim),
            # (Batch_size, vae_dim, Height/4, Width/4) ->  same
            VAE_ResidualBlock(vae_dim, vae_dim),
            # £assym (Batch_size, vae_dim, Height/4, Width/4) -> (Batch_size, 1024, Height/8, Width/8)
            nn.Conv2d(vae_dim, vae_dim, kernel_size=3, stride=2, padding=0),
            # 3X (Batch_size, vae_dim, Height/8, Width/8) -> same
            VAE_ResidualBlock(vae_dim, vae_dim),
            VAE_ResidualBlock(vae_dim, vae_dim),
            VAE_ResidualBlock(vae_dim, vae_dim),
            # (Batch_size, vae_dim, Height/8, Width/8) -> same
            VAE_AttentionBlock(vae_dim),
            # (Batch_size, vae_dim, Height/8, Width/8)-> (Batch_size, vae_dim, Height/8, Width/8)
            VAE_ResidualBlock(vae_dim, vae_dim),
            # Normalization (32 groups, vae_dim features)
            nn.GroupNorm(groupnorm, vae_dim),
            # activation : SILU (could also be RELU, but pratically SILU seems better)
            nn.SiLU(),
            # Bottleneck
            # (Batch_size, vae_dim, Height/8, Width/8) -> (Batch_size, 8, Height/8, Width/8)
            nn.Conv2d(vae_dim, latent_dim, kernel_size=3, padding=1),
            # (Batch_size, 8, Height/8, Width/8) -> same
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch_Size, Channel, Height, Width)
        noise: (Batch_Size, 4, Height/8, Width/8)
        """

        # applying an assetric padding on layers tagged # £assym above.
        # not clear why we do that, but it seems to be implemented this way in
        # AVE models re-used by stable diffusion
        # if we don't do this torch.Size([4, 3, vae_dim, vae_dim]) -> torch.Size([4, 8, 63, 63])
        # if we don't do this torch.Size([4, 3, vae_dim, vae_dim]) -> torch.Size([4, 8, 64, 64])
        for module in self:
            # print(module)
            if getattr(module, "stride", None) == (2, 2):
                # (padding left, Right, Top, Bot),
                x = F.pad(x, (0, 1, 0, 1))

            x = module(x)

        # Use torch.chunk to divive x in 2 tensors along axis 2
        # (batch_size, 8, Heigh/8, Width/8) -> (
        #                     (batch_size, 4, Heigh/8, Width/8),
        #                     (batch_size, 4, Heigh/8, Width/8)
        # )
        mean, log_variance = torch.chunk(x, 2, dim=1)

        # use clamp to maintain log_variance in a reasonable range (not too big, not to small)
        # (batch_size, 4, Heigh/8, Width/8)
        log_variance = torch.clamp(log_variance, -30, 20)

        # get variance from log_variance (by exp)
        # (batch_size, 4, Heigh/8, Width/8)
        variance = log_variance.exp()

        # compute standard deviation
        # (batch_size, 4, Heigh/8, Width/8)
        stdev = variance.sqrt()

        # To obtain a distribution X(mean, variance) starting from an other
        # distribution Z(0, 1) we can do X = mean + stdev * Z
        # Transform N(0, 1) -> N(mean, stdev)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise

        # scale the output by a constant (not clear why, historical reason ?)
        x *= self.latent_const

        return x


if __name__ == "__main__":
    h_w = 512
    batch_size = 4
    latent_dim = 8
    input_channels = 4
    vae_dim = 512  # 512
    x = torch.Tensor(batch_size, input_channels, h_w, h_w)
    noise = torch.Tensor(batch_size, int(latent_dim / 2), int(h_w / 8), int(h_w / 8))
    print(x.shape)
    model = VAE_Encoder(
        input_channels=input_channels, vae_dim=vae_dim, latent_dim=latent_dim
    )
    y = model(x, noise)
    print(y.shape)
