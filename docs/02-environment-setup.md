# 환경 구성

## GPU 확인

```bash
nvidia-smi
```

MIG에서는 물리 GPU 전체 메모리가 아니라 프로세스에 할당된 인스턴스 메모리를 기준으로 판단한다.

## Python 환경

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install diffusers transformers accelerate safetensors pillow
```

이미 PyTorch가 설치되어 있다면 CUDA 호환성을 먼저 확인하고 무작정 재설치하지 않는다.

## CUDA 점검

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))
```

## Hugging Face

모델 페이지의 사용 조건에 동의하고 필요한 경우 `huggingface-cli login`을 실행한다. 토큰은 코드나 Git에 저장하지 않는다.

## 저장공간

```bash
df -h
du -sh ~/.cache/huggingface
```
