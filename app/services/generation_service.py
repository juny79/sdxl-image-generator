"""Application service for text-to-image generation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from app.pipelines.sdxl_pipeline import SDXLPipeline


class GenerationService:
    def __init__(self, output_dir: Path = Path("outputs/generated")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = SDXLPipeline()

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int = 42,
        steps: int = 30,
        guidance_scale: float = 7.0,
        width: int = 1024,
        height: int = 1024,
    ) -> tuple[Path, Path]:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if seed < 0 or steps < 1 or width % 8 or height % 8:
            raise ValueError("invalid generation parameters")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        stem = f"{timestamp}_service_seed{seed}"
        image_path = self.output_dir / f"{stem}.png"
        metadata_path = self.output_dir / f"{stem}.json"
        image = self.pipeline.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        image.save(image_path)
        metadata_path.write_text(json.dumps({
            "model_id": self.pipeline.model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance_scale,
            "image_path": str(image_path),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return image_path, metadata_path
