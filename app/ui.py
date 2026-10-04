"""Gradio web UI for SDXL text-to-image and img2img."""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import torch

from app.services.generation_service import GenerationService
from app.services.img2img_service import Img2ImgService


def cleanup(service) -> None:
    del service
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_image(prompt, negative_prompt, seed, steps, guidance_scale):
    service = GenerationService()
    try:
        image_path, metadata_path = service.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            steps=int(steps),
            guidance_scale=float(guidance_scale),
        )
        return str(image_path), str(metadata_path)
    finally:
        cleanup(service)


def transform_image(source, prompt, negative_prompt, strength, seed, steps, guidance_scale):
    if not source:
        raise gr.Error("img2img 원본 이미지를 업로드하세요.")
    service = Img2ImgService()
    try:
        image_path, metadata_path = service.transform(
            source_path=Path(source),
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=float(strength),
            seed=int(seed),
            steps=int(steps),
            guidance_scale=float(guidance_scale),
        )
        return str(image_path), str(metadata_path)
    finally:
        cleanup(service)


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="SDXL Image Generator") as demo:
        gr.Markdown("# SDXL Image Generator\nSDXL base 1.0 기반 이미지 생성 및 img2img")
        with gr.Tab("Text-to-Image"):
            prompt = gr.Textbox(label="Prompt", value="A cozy Korean tea house in a quiet forest")
            negative = gr.Textbox(label="Negative prompt", value="blurry, low quality, text, watermark")
            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                steps = gr.Slider(1, 80, value=30, step=1, label="Steps")
                scale = gr.Slider(1, 15, value=7, step=0.5, label="Guidance scale")
            button = gr.Button("Generate", variant="primary")
            image = gr.Image(label="Result", type="filepath")
            metadata = gr.File(label="Metadata")
            button.click(generate_image, [prompt, negative, seed, steps, scale], [image, metadata])

        with gr.Tab("Img2Img"):
            source = gr.Image(label="Source image", type="filepath")
            i_prompt = gr.Textbox(label="Prompt", value="A watercolor illustration of the same scene, soft pastel colors")
            i_negative = gr.Textbox(label="Negative prompt", value="blurry, low quality, text, watermark")
            with gr.Row():
                i_strength = gr.Slider(0.05, 1.0, value=0.55, step=0.05, label="Strength")
                i_seed = gr.Number(label="Seed", value=42, precision=0)
                i_steps = gr.Slider(1, 80, value=30, step=1, label="Steps")
                i_scale = gr.Slider(1, 15, value=7, step=0.5, label="Guidance scale")
            i_button = gr.Button("Transform", variant="primary")
            i_image = gr.Image(label="Result", type="filepath")
            i_metadata = gr.File(label="Metadata")
            i_button.click(transform_image, [source, i_prompt, i_negative, i_strength, i_seed, i_steps, i_scale], [i_image, i_metadata])
    return demo


if __name__ == "__main__":
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    build_demo().launch(server_name="0.0.0.0", server_port=port)
