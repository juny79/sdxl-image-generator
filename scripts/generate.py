"""Generate one SDXL image and save its reproducibility metadata."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import torch
from diffusers import DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_PROMPT = (
    "A small traditional Korean tea house in a quiet forest, "
    "soft morning sunlight, cinematic photography, detailed wood texture"
)
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted architecture, text, watermark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with SDXL base 1.0")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generated"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.width % 8 or args.height % 8:
        raise SystemExit("width and height must be multiples of 8")
    if args.steps < 1 or args.seed < 0:
        raise SystemExit("steps must be positive and seed must not be negative")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run scripts/check_environment.py first.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{timestamp}_t2i_seed{args.seed}"
    image_path = args.output_dir / f"{stem}.png"
    metadata_path = args.output_dir / f"{stem}.json"

    print(f"Loading model: {MODEL_ID}")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    # Recent Diffusers versions expose VAE slicing on the VAE component.
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    started = time.perf_counter()
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    ).images[0]
    elapsed = time.perf_counter() - started
    image.save(image_path)

    metadata = {
        "model_id": MODEL_ID,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "elapsed_seconds": round(elapsed, 3),
        "image_path": str(image_path),
        "created_at_utc": timestamp,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved image: {image_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Generation time: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
