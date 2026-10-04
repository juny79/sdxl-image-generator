# 프로젝트 종합 계획서

## 1. 목표

`stabilityai/stable-diffusion-xl-base-1.0`을 사용해 텍스트 기반 이미지 생성 프로그램을 만든다. 사용자는 프롬프트, negative prompt, seed, steps, CFG를 지정하고 결과 이미지와 생성 조건을 저장할 수 있어야 한다.

## 2. 범위

### 포함

- 1024×1024 text-to-image
- seed·steps·guidance scale·프롬프트 비교
- img2img
- 생성 이력과 메타데이터 저장
- LoRA 적용 및 소규모 LoRA 학습
- A100 MIG 20GB 메모리 최적화

### 제외

- 전체 모델 사전학습
- 전체 가중치 미세조정
- 대규모 분산 학습
- 모델·데이터의 무단 배포

## 3. 단계별 계획

| 단계 | 결과물 | 완료 기준 |
|---|---|---|
| 1. 환경 검증 | GPU 점검 로그 | CUDA와 PyTorch가 GPU를 인식 |
| 2. 기본 추론 | 첫 생성 이미지 | 1024×1024 PNG 생성 |
| 3. 비교 실험 | 비교 이미지·JSON | 조건별 결과 재현 가능 |
| 4. img2img | strength별 결과 | 원본 유지 정도 차이 확인 |
| 5. 프로그램화 | CLI/API | 입력 검증과 파일 저장 동작 |
| 6. LoRA | LoRA 가중치 | 기본 모델과 비교 가능 |
| 7. 검증·문서화 | 테스트 보고서 | 문서만으로 재실행 가능 |

## 4. 권장 기술 스택

Python 3.10+, PyTorch, Diffusers, Transformers, Accelerate, Safetensors, Pillow, FastAPI 또는 Gradio, JSONL/SQLite, Git

## 5. 성공 기준

- 동일 모델·seed·조건에서 결과 재현
- 생성 이미지와 메타데이터 동시 저장
- MIG 20GB에서 기본 추론 성공
- OOM 발생 시 단계별 완화 절차 확보
- LoRA 적용 전후를 동일 조건으로 비교
- 라이선스와 데이터 출처 기록
