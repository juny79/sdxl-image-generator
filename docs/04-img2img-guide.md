# img2img 이미지 변환

입력 이미지의 구도와 형태를 유지하면서 프롬프트에 맞게 변환한다.

```python
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
)
pipe.to("cuda")
pipe.enable_vae_slicing()
source = Image.open("data/input/source.png").convert("RGB").resize((1024, 1024))
result = pipe(
    prompt="A watercolor illustration of the same scene, soft pastel colors",
    image=source, strength=0.55, num_inference_steps=30, guidance_scale=7.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]
result.save("outputs/generated/sdxl_img2img.png")
```

`strength`는 0.35(원본 유지), 0.55(균형), 0.75(큰 변환)를 비교한다.
