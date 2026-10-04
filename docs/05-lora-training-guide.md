# SDXL LoRA 미세조정

## 데이터

사용 권한이 있는 이미지 10~30장과 `metadata.jsonl`을 준비한다. 이미지별 캡션은 실제 내용을 설명해야 한다.

```text
data/lora/
├── 0001.png
├── 0002.png
└── metadata.jsonl
```

```json
{"file_name":"0001.png","text":"a studio photo of a red ceramic vase, soft shadow"}
```

## 학습 준비

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .
pip install -r examples/text_to_image/requirements_sdxl.txt
accelerate config
python examples/text_to_image/train_text_to_image_lora_sdxl.py --help
```

스크립트 경로와 인자는 Diffusers 버전에 따라 달라질 수 있으므로 `--help`를 기준으로 실행한다.

## 시작 설정

`train_batch_size=1`, `gradient_checkpointing`, `mixed_precision=fp16`, rank 8, 100~500 steps부터 시험한다. OOM 발생 시 해상도를 768 또는 512로 낮추고, rank·steps를 줄인다.

## 평가

기본 모델과 LoRA 모델을 동일한 프롬프트·seed·steps·CFG로 생성한다. 학습 이미지 복제 여부와 새로운 배경·구도에서의 일반화 여부를 모두 확인한다.
