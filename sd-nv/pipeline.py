import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from utils import rescale, get_time_embedding


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8


def generate(
    prompt: str,
    uncond_prompt: str,
    input_image=None,
    strength: float = 0.8,
    do_cfg: bool = True,
    cfg_scale: float = 0.75,
    sampler_name: str = "ddpm",
    n_inference_steps: int = 50,
    models: dict = {},
    seed: int = None,
    device=None,
    idle_device=None,
    tokenizer=None,
    seq_len: int = 77,
):
    """_summary_


    Args:
        prompt (str): conditional context prompt
        uncond_prompt (str): negative prompt (or empty string), will provide context to be avoided by the model
        input_image (_type_, optional): input image in case of image to image. Defaults to None.
        strength (float, optional): _description_. Defaults to 0.8.
        do_cfg (bool, optional): True => do classifier free guidance (cfg). Defaults to True.
        cfg_scale (float, optional): tells how much the model will pay attention (conversely, diverge) to the prompt and input image . Defaults to 0.75.
        sampler_name (str, optional): _description_. Defaults to "ddpm".
        n_inference_steps (int, optional): _description_. Defaults to 50.
        models (dict, optional): _description_. Defaults to {}.
        seed (_type_, optional): _description_. Defaults to None.
        device (_type_, optional): _description_. Defaults to None.
        idle_device (_type_, optional): _description_. Defaults to None.
        tokenizer (_type_, optional): _description_. Defaults to None.
        seq_len  (int, optional): number max of tokens in the sequence

    Raises:
        ValueError: _description_
    """
    with torch.no_grad():
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle: lambda x: x.to(idle_device)
        else:
            to_idle: lambda x: x

    # preparing noise generator
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)

    clip = models["clip"]
    clip.to(device)

    if do_cfg:
        # do_cfg : We do classifier free guidance
        # convert the prompt into tokens using the tokenizer
        cond_tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=seq_len
        ).input_ids
        # (batch_size, seq_len)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        cond_context = clip(cond_tokens)

        # same for negative prompt
        cond_tokens = tokenizer.batch_encode_plus(
            [uncond_prompt], padding="max_length", max_length=seq_len
        ).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
        uncond_context = clip(uncond_tokens)

        # concatenate prompts
        # -> (2, seq_len, dim) = (2, 77, 768)
        context = torch.cat([cond_context, uncond_context])
    else:
        # convert it into list of tokens
        tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=seq_len
        ).input_ids
        # (batch_size, seq_len)
        tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
        context = clip(tokens)
        # (1, 77, 768)
        context = clip(tokens)

    # put the model on idle (likely cpu) to give room on gpu  for next steps
    to_idle(clip)

    if sampler_name == "ddpm":
        sampler = DDPMSampler(generator)
        sampler.set_inference_steps(n_inference_steps)
    else:
        raise ValueError(f"Unknown sampler {sampler_name} ")

    latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

    if input_image:
        encoder = models["encoder"]
        encoder.to(device)
        # Reshaping input image to enter in format expected by VAE and Unet:
        # (b, c, h, w) where w = WIDTH, h = HEIGHT and pixels between -1 and 1
        input_image_tensor = input_image.resize((WIDTH, HEIGHT))
        input_image_tensor = np.array(input_image_tensor)
        # (h, w, 3)
        input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
        # Unet expects every pixels between -1 and +1
        input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
        # (H, W, C) -> (batch_size, h, w, c)
        input_image_tensor = input_image_tensor.unsqueeze(0)
        # (batch_size, h, w, c) -> (batch_size, c, h, w)
        input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

        # generate noise using generator instanciated above
        encoder_noise = torch.randn(latents_shape, generator=generator, device=device)

        # run the image throught the encoder of the VAE
        latents = encoder(input_image_tensor, encoder_noise)

        # ???? strength will set a time stamp schedule.
        sampler.set_strength(strength=strength)
        latents = sampler.add_noise(latents, sampler.timesteps[0])

        # we are done with the encoder -> put to idle
        to_idle(encoder)
    else:
        # no input image (text to image)-> start with random noise N(0, I)
        latents = torch.randn(latents_shape, generator=generator, device=device)

    # loading the diffusion model (modified version of unet)
    diffusion = models["diffusion"]
    diffusion.to(device)

    timesteps = tqdm(sampler.timesteps)

    # Loop of diffusion, that will remove noise from the latents a each defined timestep
    # iterate over timesteps evenly separated,
    # if total steps at training time was 1000 and timesteps= 50
    # then we will process time steps 1000 .. 980 .. every 20 step (ie 1000/50)
    for i, timestep in enumerate(timesteps):
        # (1, 320)
        time_embedding = get_time_embedding(timestep).to(device)

        # (batch_size, 4, latents_height, latent_width)
        model_input = latents

        if do_cfg:
            # (batch_size, 4, latents_height, latent_width)
            # -> # (2 * batch_size, 4, latents_height, latent_width)
            # we duplicate input to have it with conditional context
            # but ALSO with unconditional context (or negative prompt)
            model_input = model_input.repeat(2, 1, 1, 1)

        # model_output is predicted noise by the UNET
        model_output = diffusion(model_input, context, time_embedding)

        if do_cfg:
            # in that case the output is 2 * batch_size
            # merge unconditional and conditional prompts together
            # weight of unconditional effect (how much do we not pay attention to input)
            # is monitored by the cfg_scale
            output_cond, oupt_uncond = model_output.chunk(2)
            model_output = cfg_scale * (output_cond - oupt_uncond) + oupt_uncond

        # the diffusion (unet) model has predicted the level of noise that
        # was applied to the input
        # Now let's remove this noise from the latents ==> the sampler dooes this
        latents = sampler.step(timestep, latents, model_output)

    # put diffusion model to idle
    to_idle(diffusion)

    # now the latents are fully denoised, lets generate the final output
    decoder = models["decoder"]
    decoder.to(device)

    images = decoder(latents)

    # put decoder to idle
    to_idle(decoder)

    # shape image to original
    images = rescale(images, (-1, 1), (0, 255), clamp=True)
    images = images.permute(0, 2, 3, 1)
    images = images.to("cpu", torch.uint8).numpy()

    return images[0]
