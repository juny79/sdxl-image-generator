# SDXL 이미지 생성 프로그램 프로젝트 종합계획서

## 1. 프로젝트 개요

`stabilityai/stable-diffusion-xl-base-1.0`을 기반으로 사용자가 프롬프트를 입력해 이미지를 생성하고, 생성 조건을 저장·비교할 수 있는 프로그램을 개발한다.

대상 환경은 NVIDIA A100 MIG 20GB이며, 1024×1024 추론과 소규모 LoRA 실습을 기준으로 한다.

## 2. 목표

1. SDXL base 1.0 기반 text-to-image 구현
2. 프롬프트, negative prompt, seed, steps, guidance scale 입력 지원
3. 이미지와 생성 메타데이터 저장
4. 동일 조건 및 변수별 결과 비교
5. img2img 변환 지원
6. LoRA 적용 및 제한적인 소규모 학습
7. 재현 가능한 프로젝트 구조와 문서 구축

## 3. 범위

### 포함

- 1024×1024 이미지 생성
- seed·steps·CFG·prompt 비교
- img2img
- CLI 또는 API 기반 프로그램
- 생성 이력과 실험 메타데이터 관리
- A100 MIG 메모리 최적화
- LoRA 학습 및 적용 결과 검증

### 제외

- 모델 전체 사전학습
- SDXL 전체 가중치 미세조정
- 대규모 분산 학습
- 다중 사용자 상용 서비스 운영

## 4. 시스템 구성

```text
사용자
  ↓
CLI / FastAPI / Gradio
  ↓
생성 서비스
  ├── SDXL text-to-image pipeline
  ├── SDXL img2img pipeline
  ├── LoRA loader
  └── metadata writer
  ↓
outputs/ 이미지·JSON
```

## 5. 단계별 진행 계획

| 단계 | 주요 작업 | 결과물 | 완료 기준 |
|---|---|---|---|
| 1 | 저장소와 문서 구조 확정 | README·본 계획서 | 루트 문서와 docs 구조 확인 |
| 2 | GPU·CUDA·패키지 점검 | 환경 점검 로그 | CUDA 사용 가능 |
| 3 | 기본 추론 구현 | 첫 PNG | 1024×1024 생성 성공 |
| 4 | 비교 실험 구현 | 비교 이미지·JSON | seed·steps·CFG 비교 가능 |
| 5 | img2img 구현 | 변환 결과 | strength별 차이 확인 |
| 6 | 프로그램화 | CLI/API | 입력 검증·저장 동작 |
| 7 | LoRA 실습 | LoRA 가중치 | 기본 모델과 전후 비교 |
| 8 | 테스트·문서화 | 테스트 보고서 | 문서만으로 재현 가능 |

## 6. 권장 일정

- 환경 검증: 0.5~1일
- 기본 추론: 1일
- 비교·img2img: 1~2일
- CLI/API: 2~4일
- LoRA: 2~5일
- 검증·문서화: 1~2일

## 7. 메모리 운영 기준

- 추론 dtype은 `float16`을 사용한다.
- 기본 batch size는 1로 한다.
- VAE slicing을 활성화한다.
- 학습은 gradient checkpointing을 사용한다.
- OOM 발생 시 동시 GPU 작업 종료, 해상도 감소, LoRA rank 감소, steps 감소 순서로 대응한다.

## 8. 산출물

- 기본 생성 이미지 3장 이상
- seed·steps·CFG 비교 결과
- img2img strength 비교 결과
- LoRA 학습 데이터와 캡션
- LoRA 학습 로그와 가중치
- 기본 모델·LoRA 모델 동일 조건 비교
- 실험 메타데이터
- 설치·실행·문제 해결 문서

## 9. 성공 기준

- 동일 모델·seed·조건에서 재현 가능한 결과 생성
- 이미지와 메타데이터 동시 저장
- A100 MIG 20GB에서 기본 추론 완료
- 잘못된 입력값을 검증하고 오류를 설명
- LoRA 적용 전후를 동일 조건으로 비교
- 모델과 데이터의 라이선스 및 출처 기록

## 10. 버전 관리 규칙

### 파일명

`YYYYMMDD_<type>_<experiment-id>_<short-name>.<ext>`

예: `20250115_t2i_exp0001_korean-teahouse.png`

### 커밋

- `docs:` 문서 변경
- `feat:` 기능 추가
- `fix:` 오류 수정
- `test:` 테스트 변경
- 모델 가중치, 개인 데이터, 토큰, 대용량 결과물은 커밋하지 않는다.

## 11. 다음 실행 단계

1. `docs/02-environment-setup.md`를 기준으로 환경 점검
2. PyTorch의 CUDA 인식 결과 기록
3. 기본 SDXL 생성 스크립트 구현
4. 첫 이미지와 메타데이터 저장
5. 결과 확인 후 API 또는 UI 구현으로 진행
