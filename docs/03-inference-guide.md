# 기본 생성 및 프롬프트 비교

## 기준 설정

- 모델: `stabilityai/stable-diffusion-xl-base-1.0`
- 크기: 1024×1024
- dtype: `float16`
- steps: 30
- guidance scale: 7.0
- seed: 42

## 최소 생성 코드

```python
import torch
from diffusers import DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")
pipe.enable_vae_slicing()

prompt = "A small traditional Korean tea house in a quiet forest, soft morning sunlight"
negative_prompt = "blurry, low quality, distorted architecture, text, watermark"
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1024, height=1024,
    num_inference_steps=30, guidance_scale=7.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
image.save("outputs/generated/sdxl_first.png")
```

## 비교 항목

seed(11·42·2025), steps(20·30·40), guidance scale(5·7·9), 프롬프트 상세도, negative prompt를 한 번에 하나씩 변경한다. 모든 결과에 JSON 메타데이터를 저장한다.
