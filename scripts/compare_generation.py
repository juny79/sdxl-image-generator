"""Compare SDXL generation settings while changing one variable at a time."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT = (
    "A small traditional Korean tea house in a quiet forest, "
    "soft morning sunlight, cinematic photography, detailed wood texture"
)
NEGATIVE_PROMPT = "blurry, low quality, distorted architecture, text, watermark"
BASELINE = {"seed": 42, "steps": 30, "guidance_scale": 7.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SDXL parameter comparison")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/comparisons"))
    return parser.parse_args()


def generate(pipe, output_dir: Path, label: str, seed: int, steps: int, guidance_scale: float) -> dict:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    started = time.perf_counter()
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=1024,
        height=1024,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    elapsed = time.perf_counter() - started
    image_path = output_dir / f"{label}.png"
    image.save(image_path)
    return {
        "model_id": MODEL_ID,
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "seed": seed,
        "width": 1024,
        "height": 1024,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "elapsed_seconds": round(elapsed, 3),
        "image_path": str(image_path),
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run scripts/check_environment.py first.")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading model: {MODEL_ID}")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.to("cuda")
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    experiments = []
    for seed in [11, 42, 2025]:
        experiments.append((f"seed_{seed}", seed, BASELINE["steps"], BASELINE["guidance_scale"]))
    for steps in [20, 30, 40]:
        experiments.append((f"steps_{steps}", BASELINE["seed"], steps, BASELINE["guidance_scale"]))
    for scale in [5.0, 7.0, 9.0]:
        experiments.append((f"cfg_{str(scale).replace('.', '_')}", BASELINE["seed"], BASELINE["steps"], scale))

    records = []
    for index, (label, seed, steps, scale) in enumerate(experiments, start=1):
        print(f"[{index}/{len(experiments)}] {label}")
        record = generate(pipe, output_dir, label, seed, steps, scale)
        records.append(record)

    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved comparison results: {output_dir}")
    print(f"Saved manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
