import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str, uncond_prompt: str, input_image=None, strength: float=0.8,
             do_cfg: bool=True, cfg_scale: float=7.5, sample_name: str="ddpm", n_inference_steps: int=50,
            models: dict={}, seed: int=None, device=None, idle_device=None, tokenizer=None, seq_len:int =77):
    """_summary_

    Args:
        prompt (str): conditional context prompt
        uncond_prompt (str): negative prompt, will provide context to be avoided by the model 
        input_image (_type_, optional): _description_. Defaults to None.
        strength (float, optional): _description_. Defaults to 0.8.
        do_cfg (bool, optional): True = do classifier free guidance (cfg). Defaults to True.
        cfg_scale (float, optional): tells how much the model will pay attention (conversely, diverges) to the prompt and input image  . Defaults to 7.5.
        sample_name (str, optional): _description_. Defaults to "ddpm".
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
            [prompt], padding="max_length", max_length=seq_len).input_ids
        #(batch_size, seq_len)
        cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, 
                                   device=device)
        cond_context = clip(cond_tokens)
        
        # same for negative prompt
        cond_tokens = tokenizer.batch_encode_plus(
            [uncond_prompt], padding="max_length", max_length=seq_len).input_ids
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, 
                                   device=device)
        uncond_context = clip(uncond_tokens)

        # concatenate prompts
        # -> (2, seq_len, dim) = (2, 77, 768)
        context = torch.cat([cond_context, uncond_context])
    else:
        # convert it into list of tokes
        tokens = tokenizer.batch_encode_plus(
            [prompt], padding="max_length", max_length=seq_len).input_ids
        #(batch_size, seq_len)
        tokens = torch.tensor(cond_tokens, dtype=torch.long, 
                                   device=device)
        context = clip(tokens)
        # (1, 77, 768)
        context = clip(tokens)

    to_idle(clip)