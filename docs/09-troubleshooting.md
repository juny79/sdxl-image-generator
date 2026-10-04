# 문제 해결

## CUDA를 사용할 수 없음

`torch.cuda.is_available()`와 PyTorch의 CUDA 버전을 확인한다. 환경에 맞지 않는 PyTorch를 설치하지 않는다.

## CUDA out of memory

동시 작업을 종료하고 batch size 1, VAE slicing, gradient checkpointing을 적용한다. 학습은 해상도·rank·steps 순으로 낮춰 시험한다.

## 모델 다운로드 실패

Hugging Face 인증, 모델 사용 조건 동의, 네트워크와 디스크 용량을 확인한다.

## 생성이 매우 느림

MIG 할당량, CPU offload 여부, 다른 GPU 프로세스, 모델 재로딩 여부를 확인한다.

## 결과 재현 불가

모델 버전, Diffusers 버전, seed, prompt, steps, CFG, dtype와 Git commit을 함께 기록한다.
