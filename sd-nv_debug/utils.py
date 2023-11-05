import torch


def rescale(x: torch.tensor, old_range: tuple, new_range: tuple, clamp: bool = False):
    """Will put input image pixels values between suited ranges (-1, 1) or opposite transformation (0, 255)

    Args:
        x (torch.tensor): input images (batch_size, channel, w, h)
        old_range (tuple): (0, 255) : before rescale (reverse will be (-1, 1))
        new_range (tuple): (-1, 1) : before rescale (reverse will be (0, 255))
        clamp (bool, optional): limit computed values to new ranges boundaries. Defaults to False.
    """
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x


def get_time_embedding(timestep: int, dim: int = 320) -> torch.tensor:
    """Will transform the input timestep (likely an integer between 0 and 1000)
    into an embedding in shape [1, dim] that can be processed as a context of time
    (ie number of denoising iterations) by the diffusion model
    This embedding is computed in a similar way than positionnal embeddings of token in a sentence

    Embedding[i, 2k]   = sin(position / (10000^(2k   / dim)))
    Embedding[i, 2k+1] = cos(position / (10000^(2k+1 / dim)))

    Args:
        timestep (int): value between 0 and 1000 that is the number of denoising operation (inference) or noising operation (training) that were applied on the latents input.
        dim (int, optional): dimension of the positionnal embedding vectore. Defaults to 320.

    Returns:
        torch.tensor: time embedding of dim (1, dim) (in the paper : 1, 320)
    """

    # (160, )
    freqs = torch.pow(
        10000, -torch.arange(start=0, end=int(dim / 2), dtype=torch.float32)
    )
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
