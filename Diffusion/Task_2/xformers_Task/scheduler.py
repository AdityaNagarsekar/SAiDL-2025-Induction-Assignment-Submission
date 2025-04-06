# scheduler.py
"""
Defines a dummy scheduler for benchmarking purposes.
Does not perform actual diffusion steps.
"""
import torch
from typing import Dict, Any, List

class DummyScheduler:
    """
    A placeholder scheduler that mimics the interface of a diffusion scheduler
    (like DDPM or DDIM) for benchmarking the model's forward pass within a
    sampling loop.

    It provides `set_timesteps` and `step` methods but does not implement
    any actual diffusion noise scheduling or denoising logic.

    Args:
        num_steps (int): The initial number of timesteps to configure.
    """
    def __init__(self, num_steps: int):
        # Initialize with a simple linear sequence of timesteps (descending)
        self.timesteps: List[int] = list(range(num_steps, 0, -1))
        self.num_inference_steps: int = num_steps

    def set_timesteps(self, num_inference_steps: int):
        """
        Sets the number of timesteps for the inference loop.

        Args:
            num_inference_steps (int): The desired number of simulation steps.
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = list(range(num_inference_steps, 0, -1))

    def step(self, noise_pred: torch.Tensor, timestep: int, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Simulates a scheduler step.

        Performs an arbitrary calculation involving the inputs to mimic the
        computational graph dependencies of a real scheduler step, but does
        not implement any diffusion logic.

        Args:
            noise_pred (torch.Tensor): The model's noise prediction output.
            timestep (int): The current timestep. (Not actually used in this dummy version).
            latents (torch.Tensor): The current 'latents' (input to the model for this step).

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing the 'previous sample'
                                     (a modified version of the input latents).
        """
        # Perform a simple, arbitrary operation to simulate work and graph connection.
        # This does NOT reflect any real diffusion sampling algorithm.
        prev_latents = latents - noise_pred * 0.01 # Example dummy calculation
        return {"prev_sample": prev_latents}

    def __len__(self):
        return self.num_inference_steps