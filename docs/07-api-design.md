# 프로그램 구조 및 API 설계

## 권장 구조

- `app/pipelines/`: Diffusers 파이프라인 로딩·재사용
- `app/services/`: 생성·img2img·메타데이터 저장
- `app/main.py`: FastAPI 또는 Gradio 진입점
- `scripts/`: 환경 점검과 배치 실행
- `outputs/`: 생성 결과와 로그

## 최소 API

`POST /generate`는 prompt, negative_prompt, seed, width, height, steps, guidance_scale을 입력받고 image_path와 metadata_path를 반환한다.

`POST /img2img`는 source_path, prompt, strength와 공통 생성 옵션을 입력받는다.

## 구현 원칙

모델은 요청마다 다시 로드하지 않고 프로세스 시작 시 한 번 로드한다. 동시 요청은 GPU 메모리를 고려해 기본적으로 1개로 제한한다. 입력값은 해상도, steps, CFG, strength 범위를 검증한다.
