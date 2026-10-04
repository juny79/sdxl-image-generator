"""Application service for SDXL img2img generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"


class Img2ImgService:
    def __init__(self, output_dir: Path = Path("outputs/img2img")) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        self.pipe.to("cuda")
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

    def transform(
        self,
        source_path: Path,
        prompt: str,
        negative_prompt: str,
        strength: float = 0.55,
        seed: int = 42,
        steps: int = 30,
        guidance_scale: float = 7.0,
    ) -> tuple[Path, Path]:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not prompt.strip() or not 0.0 < strength <= 1.0 or seed < 0 or steps < 1:
            raise ValueError("invalid img2img parameters")

        source = Image.open(source_path).convert("RGB").resize((1024, 1024))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        label = str(strength).replace(".", "_")
        stem = f"{timestamp}_img2img_strength{label}_seed{seed}"
        image_path = self.output_dir / f"{stem}.png"
        metadata_path = self.output_dir / f"{stem}.json"
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=source,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        ).images[0]
        image.save(image_path)
        metadata_path.write_text(json.dumps({
            "model_id": MODEL_ID,
            "source_path": str(source_path),
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "seed": seed,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "image_path": str(image_path),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return image_path, metadata_path
