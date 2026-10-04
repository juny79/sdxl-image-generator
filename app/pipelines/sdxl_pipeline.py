"""SDXL pipeline loading utilities."""

from __future__ import annotations

import torch
from diffusers import DiffusionPipeline


class SDXLPipeline:
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0") -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.model_id = model_id
        self.pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.to("cuda")
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

    def generate(self, **kwargs):
        return self.pipe(**kwargs).images[0]
