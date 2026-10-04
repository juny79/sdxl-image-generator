"""Command-line entry point for SDXL text-to-image and img2img."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.services.generation_service import GenerationService
from app.services.img2img_service import Img2ImgService


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="blurry, low quality, text, watermark")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="SDXL image generation service")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="text-to-image generation")
    add_common_arguments(generate_parser)

    img2img_parser = subparsers.add_parser("img2img", help="image-to-image generation")
    add_common_arguments(img2img_parser)
    img2img_parser.add_argument("--source", type=Path, required=True)
    img2img_parser.add_argument("--strength", type=float, default=0.55)

    args = parser.parse_args()
    if args.command == "generate":
        image_path, metadata_path = GenerationService().generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
    else:
        image_path, metadata_path = Img2ImgService().transform(
            source_path=args.source,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            strength=args.strength,
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
    print(f"Saved image: {image_path}")
    print(f"Saved metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
