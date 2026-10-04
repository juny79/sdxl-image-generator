"""Run SDXL img2img strength comparison."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = "A watercolor illustration of the same scene, soft pastel colors, delicate paper texture"
NEGATIVE_PROMPT = "blurry, low quality, distorted, text, watermark"


def find_latest_source() -> Path:
    candidates = sorted(Path("outputs/generated").glob("*.png"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise SystemExit("No source PNG found. Use --source path/to/source.png.")
    return candidates[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDXL img2img strength comparison")
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/img2img"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run scripts/check_environment.py first.")

    source_path = args.source or find_latest_source()
    if not source_path.exists():
        raise SystemExit(f"Source image does not exist: {source_path}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(source_path).convert("RGB").resize((1024, 1024))

    print(f"Loading model: {MODEL_ID}")
    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    records = []
    for strength in [0.35, 0.55, 0.75]:
        label = f"strength_{str(strength).replace('.', '_')}"
        print(f"Generating {label}")
        started = time.perf_counter()
        result = pipe(
            prompt=args.prompt,
            negative_prompt=NEGATIVE_PROMPT,
            image=source,
            strength=strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]
        elapsed = time.perf_counter() - started
        image_path = output_dir / f"{label}.png"
        result.save(image_path)
        records.append({
            "model_id": MODEL_ID,
            "source_path": str(source_path),
            "prompt": args.prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": args.seed,
            "strength": strength,
            "num_inference_steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "elapsed_seconds": round(elapsed, 3),
            "image_path": str(image_path),
        })

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved img2img results: {output_dir}")
    print(f"Saved manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
