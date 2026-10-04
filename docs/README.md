# SDXL 이미지 생성 프로그램 문서

Stable Diffusion XL base 1.0(`stabilityai/stable-diffusion-xl-base-1.0`) 기반 프로젝트의 설계, 구현, 실험, 운영 문서다.

## 기준 환경

- GPU: NVIDIA A100 MIG 20GB
- 기본 해상도: 1024×1024
- 추론 dtype: `float16`
- 범위: text-to-image, 프롬프트 비교, img2img, 소규모 LoRA
- 제외: 전체 모델 사전학습 및 전체 가중치 미세조정

## 진행 순서

1. [01-project-plan.md](./01-project-plan.md) — 목표·범위·일정·완료 기준
2. [02-environment-setup.md](./02-environment-setup.md) — GPU·Python·패키지 점검
3. [03-inference-guide.md](./03-inference-guide.md) — 기본 생성·파라미터 비교
4. [04-img2img-guide.md](./04-img2img-guide.md) — 이미지 변환
5. [05-lora-training-guide.md](./05-lora-training-guide.md) — LoRA 학습
6. [06-experiment-management.md](./06-experiment-management.md) — 메타데이터·버전 관리
7. [07-api-design.md](./07-api-design.md) — 프로그램 구조와 API
8. [08-testing-and-evaluation.md](./08-testing-and-evaluation.md) — 테스트·평가
9. [09-troubleshooting.md](./09-troubleshooting.md) — 장애 대응
10. [10-license-and-security.md](./10-license-and-security.md) — 라이선스·보안

## 구현 원칙

- 먼저 추론을 검증한 뒤 API와 UI를 구현한다.
- 모델·프롬프트·seed·생성 파라미터를 모든 결과와 함께 기록한다.
- 모델 가중치와 개인 데이터를 Git에 커밋하지 않는다.
- 문서와 설정은 버전이 바뀌어도 재현 가능하도록 명시한다.


## `docs/02-environment-setup.md`

```markdown docs/02-environment-setup.md
# 환경 구성

## 1. GPU 확인

```bash
nvidia-smi
```

MIG 환경에서는 할당된 GPU 인스턴스와 프로세스가 사용할 수 있는 메모리를 확인한다. 물리 GPU 전체 메모리와 실제 할당 메모리는 다를 수 있다.

## 2. Python 환경

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install torch torchvision
pip install diffusers transformers accelerate safetensors pillow
```

설치된 CUDA 환경에 맞는 PyTorch를 사용해야 한다. 기존 환경에 PyTorch가 설치되어 있다면 먼저 다음을 확인한다.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

## 3. CUDA 확인

```python
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print(
        "VRAM GiB:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1),
    )
```

## 4. Hugging Face 인증

모델 접근 시 라이선스 동의 또는 인증이 필요한 경우 다음을 실행한다.

```bash
huggingface-cli login
```

토큰은 소스 코드에 직접 저장하지 않는다.

## 5. 저장공간

모델 캐시, LoRA 체크포인트, 생성 이미지가 저장되므로 다음을 정기적으로 확인한다.

```bash
df -h
du -sh ~/.cache/huggingface
```
```

## `docs/03-inference-guide.md`

```markdown docs/03-inference-guide.md
# 기본 이미지 생성

## 1. 생성 기준

- 모델: `stabilityai/stable-diffusion-xl-base-1.0`
- 해상도: 1024×1024
- dtype: `float16`
- steps: 30
- guidance scale: 7.0
- seed: 42

## 2. 기본 코드

```python
import torch
from diffusers import DiffusionPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

pipe.to("cuda")
pipe.enable_vae_slicing()

prompt = (
    "A small traditional Korean tea house in a quiet forest, "
    "soft morning sunlight, cinematic photography, detailed wood texture"
)
negative_prompt = "blurry, low quality, distorted architecture, text, watermark"

generator = torch.Generator(device="cuda").manual_seed(42)

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=1024,
    height=1024,
    num_inference_steps=30,
    guidance_scale=7.0,
    generator=generator,
).images[0]

image.save("outputs/generated/sdxl_first.png")
```

## 3. 비교 실험

한 번에 하나의 변수만 바꾼다.

- seed: 11, 42, 2025
- steps: 20, 30, 40
- guidance scale: 5, 7, 9
- prompt 상세도
- negative prompt

```python
for seed in [11, 42, 2025]:
    generator = torch.Generator(device="cuda").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1024,
        height=1024,
        num_inference_steps=30,
        guidance_scale=7.0,
        generator=generator,
    ).images[0]

    image.save(f"outputs/comparisons/sdxl_seed_{seed}.png")
```

## 4. 주의사항

- 같은 결과를 재현하려면 모델 버전, seed, prompt, steps, CFG를 모두 기록한다.
- 이미지 파일명에 날짜와 실험 번호를 포함한다.
- 생성 결과와 함께 JSON 메타데이터를 저장한다.
```

## `docs/04-img2img-guide.md`

```markdown docs/04-img2img-guide.md
# img2img 이미지 변환

## 1. 목적

입력 이미지의 구도와 형태를 어느 정도 유지하면서 새로운 스타일이나 장면으로 변환한다.

## 2. 실행 예시

```python
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline

MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
)

pipe.to("cuda")
pipe.enable_vae_slicing()

init_image = Image.open("data/input/source.png").convert("RGB")
init_image = init_image.resize((1024, 1024))

result = pipe(
    prompt="A watercolor illustration of the same scene, soft pastel colors",
    image=init_image,
    strength=0.55,
    num_inference_steps=30,
    guidance_scale=7.0,
    generator=torch.Generator(device="cuda").manual_seed(42),
).images[0]

result.save("outputs/generated/sdxl_img2img.png")
```

## 3. strength 비교

| strength | 특징 |
|---:|---|
| 0.35 | 원본 구도와 형태를 많이 유지 |
| 0.55 | 원본과 변환의 균형 |
| 0.75 | 스타일과 내용 변화가 큼 |

`strength`가 높을수록 원본 이미지에서 멀어진다.
```

## `docs/05-lora-training-guide.md`

```markdown docs/05-lora-training-guide.md
# SDXL LoRA 미세조정

## 1. 목적

기본 모델 전체를 재학습하지 않고, 특정 대상이나 스타일을 표현하는 추가 가중치만 학습한다.

A100 MIG 20GB에서는 소규모 실험을 기준으로 하며, OOM이 발생할 수 있으므로 짧은 시험 학습부터 수행한다.

## 2. 데이터 권장 기준

- 이미지 수: 10~30장
- 사용 권한이 있는 이미지 사용
- 대상이 명확하게 보이는 이미지 사용
- 워터마크와 불필요한 텍스트 제거
- 이미지별 실제 내용을 설명하는 캡션 작성
- 학습용·검증용 프롬프트를 분리

예시:

```text
data/lora/
├── 0001.png
├── 0002.png
├── 0003.png
└── metadata.jsonl
```

```json
{"file_name":"0001.png","text":"a studio photo of a red ceramic vase, soft shadow"}
{"file_name":"0002.png","text":"a red ceramic vase on a wooden table, natural light"}
{"file_name":"0003.png","text":"close-up photo of a red ceramic vase, neutral background"}
```

## 3. 학습 전 확인

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers

pip install -e .
pip install -r examples/text_to_image/requirements_sdxl.txt

accelerate config
python examples/text_to_image/train_text_to_image_lora_sdxl.py --help
```

Diffusers 버전에 따라 스크립트 경로와 인자 이름이 달라질 수 있으므로 반드시 `--help` 결과를 확인한다.

## 4. 시작 설정

```bash
accelerate launch examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir="/absolute/path/to/data/lora" \
  --caption_column="text" \
  --output_dir="./outputs/lora/exp-0001" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate=1e-4 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --rank=8 \
  --checkpointing_steps=100
```

## 5. OOM 대응 순서

1. 배치 크기를 1로 유지한다.
2. gradient checkpointing을 활성화한다.
3. 해상도를 768 또는 512로 낮춰 테스트한다.
4. 동시에 실행 중인 GPU 작업을 종료한다.
5. LoRA rank를 4 또는 8로 낮춘다.
6. 학습 스텝을 100~200으로 줄여 시험한다.

## 6. 과적합 점검

다음 현상이 나타나면 과적합 가능성이 있다.

- 학습 이미지와 거의 동일한 이미지가 반복됨
- 새로운 구도와 배경에서 대상 특징이 유지되지 않음
- 특정 프롬프트에서만 결과가 정상임

기본 모델과 LoRA 모델을 동일한 seed와 프롬프트로 비교한다.