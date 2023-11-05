import torch
import numpy as np


class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_training_steps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        """_summary_

            Variances applied at each steps are computed between variance start and variance end in a markov chain computed by the scheduler

        Args:
            generator (torch.Generator): _description_
            num_training_steps (int, optional): _description_. Defaults to 1000.
            beta_start (float, optional): Variance of the first gaussian noise distribution applied to the input (at training, would be last at inference). Defaults to 0.00085.
            beta_end (float, optional):  Variance of the first gaussian noise distribution applied to the input (at training, would be last at inference). Defaults to 0.0120.
        """

        # computing all betas values at each num_training_step
        # ? named scale linear schedule in hugging face
        self.betas = (
            torch.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_training_steps,
                dtype=torch.float32,
            )
            ** 2
        )
        self.generator = generator
        self.num_training_steps = num_training_steps

        # compute alpha, which is 1 - beta
        # the alphas are needed to compute how much cumulated noise was applied
        # at a given time step (fomula in the paper)
        self.alphas = 1.0 - self.betas
        # alpha_cumprod -> [alpha_0, alpha_0 * alpha_1, alpha_1 * alpha_2, ..., alpha_n-1 * alpha_n]
        self.alpha_cumprod = torch.cumprod(self.alphas, 0)
        self.one = torch.tensor(1.0)

        # self.timesteps = torch.from_numpy(
        #     np.arange(0, num_training_steps)[::-1].copy())
        # set timesteps, default is the total range of steps as in training (1000)
        self.set_inference_timesteps(num_inference_steps=num_training_steps)

    def set_inference_timesteps(self, num_inference_steps: int = 50):
        """will set self.timetseps with a list of all inference steps
            # 999, 998, 997, ...  0 for 1000 num_inference_steps
            # 999, 999 - 20 , 999 - 40, ...  0 for 50 steps
        Args:
            num_inference_steps (int, optional): how many inference steps, usually 1000 for training, and user defined (generally 50) at inference. Defaults to 50.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy()
        )
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        """return the previous time step of provided timestep
        given self.num_inference_steps and self.num_inferencen_steps
        Ex : if timestep = 960, self.num_inference_steps=1000 and self.num_inferencen_steps=50
        then retunr will be 940 (ie 960 - (1000//50))

        Args:
            timestep (int): _description_

        Returns:
            int: _description_
        """
        prev_t = timestep - (self.num_training_steps // self.num_inference_steps)
        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)
        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t
        # clamp and be sure it doesn't reach to 0
        variance = torch.clamp(variance, min=1e-20)
        return variance

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        """Remove predicted noise (model_output) from latents

        Args:
            timestep (int): _description_
            latents (torch.Tensor): _description_
            model_output (torch.Tensor): predicted noise of the unet model
        """
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_cumprod[timestep]
        alpha_prod_t_prev = self.alpha_cumprod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        # https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=15812s
        # computing predicted original sample (named xO) of formula (15) of DDPM paper
        pred_original_sample = (
            latents - beta_prod_t**0.5 * model_output
        ) / alpha_prod_t**0.5

        # compute the coef for pred_original_sample and current sample x_t
        pred_original_sample_coeff = (
            alpha_prod_t_prev**0.5 * current_beta_t
        ) / beta_prod_t**0.5
        current_sample_coeff = current_alpha_t**0.5 * beta_prod_t_prev / beta_prod_t

        # compute the predicted previous sample mean
        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )
            variance = (self._get_variance(t) ** 0.5) * noise

        # N(0, 1) --> N(mu, sigma^2)
        # X = mu + sigma * Z, where Z ~ N(0, 1)
        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def set_strength(self, strength=1):
        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def add_noise(
        self, original_samples: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        #
        alpha_cumprod = self.alpha_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        timesteps = timesteps.to(original_samples.device)
        sqrt_alpha_prod = alpha_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()

        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        # one minus alpha bar
        # standard deviation
        sqrt_one_minus_alpha_prod = (1 - alpha_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod.flatten()

        while len(sqrt_one_minus_alpha_prod.shape) < len(original_samples.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noise = torch.randn(
            original_samples.shape,
            generator=self.generator,
            device=original_samples.device,
            dtype=original_samples.dtype,
        )
        # https://www.youtube.com/watch?v=ZBKpAp_6TGI&t=15108s
        # according to equation (4) of the DDPM paper
        # X = mean + stedv * Z
        noisy_samples = (sqrt_alpha_prod * original_samples) + (
            sqrt_one_minus_alpha_prod
        ) * noise

        return noisy_samples
