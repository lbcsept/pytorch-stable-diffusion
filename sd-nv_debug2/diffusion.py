import torch
from torch import nn
from torch.nn import functional as F
from unet import UNET, UNET_OutputLayer


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int = 320):
        """trainable module that provide the context of time
        to the diffusion model

        Args:
            n_embd (int, optional): time dime . Defaults to 320.
        """
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (1, time_dim)
        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        # (1, 4 * 380 = 1280)
        return x


class Diffusion(nn.Module):
    def __init__(self, time_dim=320, latent_dim: int = 8):
        super().__init__()
        self.time_embedding = TimeEmbedding(time_dim)
        self.latent_dim = latent_dim
        self.unet = UNET()
        self.final = UNET_OutputLayer(time_dim, int(latent_dim / 2))

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        # latent: (batch_size, 4, h/8, w/8)
        # context: (batch_size, seq_len, dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (batch, 4, h/8, w/8) -> (batch, time_dim, h/8, w/8)
        output = self.unet(latent, context, time)

        # put in final size of unet
        # (batch, time_dim, h/8, w/8) -> (batch, 4, h/8, w/8)
        output = self.final(output)

        # (batch_size, 4, h/8, w/8)
        return output


if __name__ == "__main__":
    bs = 4  # batch size
    w, h = 512, 512
    lat_dim = int(8 / 2)

    latents = torch.Tensor(bs, lat_dim, w, h)
    context = torch.Tensor(
        bs,
        lat_dim,
    )

    model = Diffusion()

    model(
        latents,
    )
