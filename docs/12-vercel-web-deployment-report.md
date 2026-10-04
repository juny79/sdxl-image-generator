# SDXL 웹서비스 고도화 및 Vercel 배포 종합보고서

## 1. 보고서 목적

현재 NVIDIA A100 MIG 20GB 인스턴스에서 실행 중인 SDXL 이미지 생성 프로그램을 웹서비스 형태로 고도화하고, Vercel을 활용해 사용자 인터페이스를 배포하는 방안을 정리한다.

핵심 결론은 다음과 같다.

> Vercel에는 GPU가 없으므로 SDXL 모델과 이미지 생성 작업을 직접 배포하지 않는다. Vercel은 프론트엔드와 API 중계 계층으로 사용하고, SDXL 추론은 A100 MIG 서버의 FastAPI 백엔드에서 수행한다.

## 2. 목표 아키텍처

```text
사용자 브라우저
        ↓ HTTPS
Vercel Next.js 프론트엔드
        ↓ HTTPS API
SDXL FastAPI 백엔드
        ↓
A100 MIG 20GB + Diffusers
        ↓
이미지 저장소 / 메타데이터 DB
```

### 구성 요소

| 구성 요소 | 권장 기술 | 역할 |
|---|---|---|
| 웹 프론트엔드 | Next.js, TypeScript, Tailwind CSS | 화면·입력·결과 갤러리 |
| 프론트엔드 배포 | Vercel | 정적·SSR 웹 배포 |
| GPU API | FastAPI, Uvicorn | 생성 요청 접수·상태 조회 |
| 추론 서버 | PyTorch, Diffusers | SDXL text-to-image·img2img |
| GPU | NVIDIA A100 MIG 20GB | 실제 이미지 생성 |
| 작업 큐 | Redis + worker 또는 단일 큐 | 동시 요청 제어 |
| 이미지 저장 | S3 호환 스토리지 | 생성 결과와 원본 저장 |
| 메타데이터 | PostgreSQL 또는 SQLite | 생성 조건·상태 저장 |
| 인증 | Vercel Auth 또는 JWT | 사용자 접근 제어 |

## 3. Vercel에 GPU 코드를 직접 배포하지 않는 이유

Vercel Functions는 일반적인 웹 요청 처리용 실행 환경이다. SDXL 전체 모델을 로드하고 GPU 추론을 실행하는 용도로 사용할 수 없다.

직접 배포할 경우 다음 문제가 발생한다.

- GPU를 사용할 수 없음
- 서버리스 함수의 실행 시간 제한
- 7GB 이상의 모델 로딩 비용과 콜드 스타트
- 함수 인스턴스 간 모델 캐시 공유 불가
- 대용량 이미지와 작업 상태 관리에 부적합

따라서 다음처럼 역할을 분리해야 한다.

- Vercel: 웹 UI, 사용자 요청, 결과 조회
- Elice A100: 모델 로딩, GPU 추론, LoRA 적용

## 4. 웹 UI 고도화 방향

### 4.1 메인 화면

- 프롬프트 입력 영역
- Negative prompt 입력 영역
- Text-to-Image / Img2Img 모드 선택
- 이미지 크기 선택
- Seed 입력 및 랜덤 seed 버튼
- Steps 슬라이더
- Guidance scale 슬라이더
- Img2Img strength 슬라이더
- LoRA 선택 및 weight 조절
- 생성 버튼
- 작업 상태 표시
- 결과 이미지 다운로드 버튼

### 4.2 결과 화면

- 생성 이미지 크게 표시
- 생성 시간 표시
- 사용 모델 표시
- 프롬프트와 파라미터 표시
- 원본·결과 이미지 비교
- 같은 조건으로 재생성
- 결과 즐겨찾기
- 메타데이터 JSON 다운로드

### 4.3 실험 비교 화면

- 여러 seed 결과를 카드 형태로 표시
- steps 또는 guidance scale 비교
- img2img strength 비교
- 동일 프롬프트 결과 묶음 관리
- 결과별 평가 점수와 메모 저장

### 4.4 UX 원칙

- 생성 중 버튼 비활성화
- 예상 대기 시간 표시
- 실패 원인 사용자 친화적 표시
- 생성 작업을 새로고침해도 잃지 않도록 작업 ID 사용
- 모바일 화면에서도 입력과 결과 확인 가능

## 5. 권장 저장소 구조

```text
frontend/
├── app/
│   ├── page.tsx
│   ├── generate/page.tsx
│   ├── history/page.tsx
│   └── api/
├── components/
│   ├── PromptForm.tsx
│   ├── GenerationProgress.tsx
│   ├── ResultCard.tsx
│   └── ParameterPanel.tsx
├── lib/
│   ├── api.ts
│   └── validation.ts
├── public/
├── package.json
└── vercel.json

backend/
├── app/
│   ├── main.py
│   ├── api/routes.py
│   ├── services/generation_service.py
│   ├── workers/gpu_worker.py
│   └── schemas.py
├── requirements.txt
└── Dockerfile
```

현재 프로젝트의 `app/services/`와 `app/pipelines/`는 GPU 백엔드로 재사용한다.

## 6. 백엔드 API 설계

### 6.1 상태 확인

```text
GET /health
```

응답 예시:

```json
{
  "status": "ok",
  "gpu": "NVIDIA A100 80GB PCIe MIG 2g.20gb",
  "cuda_available": true,
  "model_loaded": true
}
```

### 6.2 text-to-image 요청

```text
POST /v1/generations
Content-Type: application/json
```

요청 예시:

```json
{
  "mode": "text-to-image",
  "prompt": "A traditional Korean tea house in a quiet pine forest",
  "negative_prompt": "blurry, low quality, text, watermark",
  "seed": 42,
  "steps": 30,
  "guidance_scale": 7.0,
  "width": 1024,
  "height": 1024
}
```

응답 예시:

```json
{
  "job_id": "job_01JEXAMPLE",
  "status": "queued"
}
```

### 6.3 상태 조회

```text
GET /v1/generations/{job_id}
```

응답 예시:

```json
{
  "job_id": "job_01JEXAMPLE",
  "status": "completed",
  "progress": 100,
  "image_url": "https://storage.example.com/results/job_01JEXAMPLE.png",
  "metadata_url": "https://storage.example.com/results/job_01JEXAMPLE.json"
}
```

상태 값:

```text
queued → running → completed
                 ↘ failed
```

### 6.4 img2img 요청

파일 업로드는 `multipart/form-data`로 처리한다.

```text
POST /v1/img2img
```

필드:

- `image`
- `prompt`
- `negative_prompt`
- `strength`
- `seed`
- `steps`
- `guidance_scale`

## 7. 비동기 작업 처리가 필요한 이유

이미지 생성은 수 초에서 수십 초가 걸릴 수 있다. Vercel 요청을 생성 작업이 끝날 때까지 계속 열어두면 타임아웃과 재시도 문제가 발생한다.

권장 흐름:

1. Vercel이 백엔드에 생성 요청
2. 백엔드가 `job_id` 반환
3. GPU worker가 큐에서 작업 수신
4. 프론트엔드가 2~3초 간격으로 상태 조회
5. 완료 후 이미지 URL 표시

초기 실습에서는 Redis 없이 FastAPI 내부 단일 작업 큐로 시작할 수 있다. 사용자가 늘어나면 Redis와 별도 GPU worker를 도입한다.

## 8. GPU 백엔드 운영 방식

### 개발 단계

- Elice 인스턴스에서 FastAPI 실행
- SSH 터널로 로컬 테스트
- Gradio와 API를 병행해 기능 검증

### 외부 Vercel 연동 단계

SSH 터널은 개발용으로만 사용한다. Vercel 서버가 Elice의 `localhost`에 접근할 수 없기 때문이다.

필요한 조건:

- 외부에서 접근 가능한 HTTPS 백엔드 주소
- 고정 도메인 또는 안정적인 터널 주소
- TLS 인증서
- API 인증 토큰
- CORS 설정
- 방화벽과 포트 접근 정책

권장 배포 선택지:

1. GPU 인스턴스에 도메인과 HTTPS reverse proxy 구성
2. Cloudflare Tunnel 등 안정적인 터널 사용
3. GPU 추론 서버를 별도 클라우드 서비스에 배포
4. 관리형 GPU inference endpoint 사용

개발용 SSH 포트 포워딩은 개인 PC 브라우저 테스트에는 적합하지만 Vercel production 백엔드 주소로 사용할 수 없다.

## 9. Vercel 프론트엔드 배포 절차

### 9.1 프론트엔드 생성

```bash
npx create-next-app@latest frontend
cd frontend
npm install
```

권장 선택:

- TypeScript: Yes
- ESLint: Yes
- Tailwind CSS: Yes
- App Router: Yes
- src directory: Yes

### 9.2 환경 변수

Vercel Project Settings에 다음 변수를 등록한다.

```env
NEXT_PUBLIC_API_BASE_URL=https://api.example.com
NEXT_PUBLIC_APP_NAME=SDXL Image Generator
```

비밀 API 키는 `NEXT_PUBLIC_` 접두사를 사용하지 않는다.

```env
SDXL_API_TOKEN=server_only_token
```

브라우저에 노출되어도 되는 값과 서버 전용 값을 구분한다.

### 9.3 배포

```bash
npm install -g vercel
vercel login
vercel
vercel --prod
```

또는 GitHub 저장소를 Vercel에 연결해 `main` push마다 자동 배포한다.

## 10. CORS 및 보안

GPU 백엔드는 Vercel 도메인만 허용한다.

```text
https://sdxl-image-generator.vercel.app
```

보안 필수 항목:

- API 토큰 또는 JWT 인증
- 허용 origin 제한
- prompt 길이 제한
- 이미지 파일 크기 제한
- 허용 확장자 검사
- 요청당 최대 steps 제한
- 해상도 허용 목록
- 사용자별 rate limit
- 업로드 파일 바이러스 검사
- 생성 결과 자동 만료 정책
- 토큰과 모델 경로 로그 제외

사용자 입력은 프롬프트라 하더라도 로그와 데이터베이스에 무제한 저장하지 않는다.

## 11. 파라미터 서버 검증

프론트엔드 검증만으로는 충분하지 않다. 백엔드에서도 다음을 검증해야 한다.

| 항목 | 권장 범위 |
|---|---:|
| width | 512, 768, 1024 |
| height | 512, 768, 1024 |
| steps | 10~50 |
| guidance scale | 1~12 |
| strength | 0.05~0.95 |
| prompt 길이 | 프로젝트 정책에 따라 제한 |
| 동시 작업 | MIG당 1개부터 시작 |

## 12. 저장소와 결과 URL

로컬 파일 시스템은 서버 재시작이나 인스턴스 교체 시 사라질 수 있다. 운영 서비스에서는 생성 결과를 S3 호환 객체 저장소에 업로드한다.

권장 메타데이터:

- job_id
- user_id
- model_id
- prompt
- negative_prompt
- seed
- steps
- guidance_scale
- strength
- width
- height
- created_at
- status
- image_url
- metadata_url
- error_message

개인정보와 원본 이미지의 보존 기간을 별도로 정한다.

## 13. 성능 최적화

### GPU

- 모델을 프로세스 시작 시 한 번만 로드
- FP16 사용
- VAE slicing 또는 tiling 사용
- 동시 추론 1개로 제한
- 이미지 크기 제한
- 작업 큐 적용
- 생성 완료 후에만 결과 업로드

### 프론트엔드

- 생성 전 미리보기와 설정 검증
- polling 간격 2~3초
- 이미지 lazy loading
- 썸네일과 원본 분리
- 결과 목록 pagination
- 생성 버튼 debounce

### 결과 저장

- PNG와 WebP 선택 지원
- 원본은 고품질, 목록은 썸네일 사용
- 오래된 결과 자동 삭제
- 메타데이터와 이미지를 동일 job_id로 연결

## 14. 배포 단계별 계획

### 1단계: 현재 구조 안정화

- CLI와 Gradio 오류 수정
- text-to-image·img2img 결과 검증
- 메타데이터 형식 고정
- GPU 백엔드 서비스 계층 정리

### 2단계: FastAPI 전환

- `/health` 구현
- `/v1/generations` 구현
- `/v1/generations/{job_id}` 구현
- 단일 GPU 작업 큐 구현
- CORS와 API 토큰 적용

### 3단계: Next.js UI

- PromptForm
- ParameterPanel
- GenerationProgress
- ResultCard
- History 페이지
- API client

### 4단계: Vercel 배포

- GitHub repository 연결
- 환경변수 등록
- Preview deployment 검증
- Production domain 설정
- 백엔드 HTTPS 연결

### 5단계: 운영 고도화

- Redis queue
- PostgreSQL metadata
- S3 이미지 저장
- 사용자 인증
- rate limit
- 모니터링과 에러 추적

## 15. 테스트 계획

### 기능 테스트

- Vercel에서 API 요청 성공
- text-to-image 생성
- img2img 업로드
- 잘못된 파라미터 거부
- 생성 중 상태 표시
- 완료 이미지 표시
- 메타데이터 다운로드

### 장애 테스트

- GPU 작업 중복 요청
- 백엔드 연결 실패
- 생성 중 프로세스 재시작
- 이미지 업로드 실패
- 모델 로딩 실패
- Vercel 새로고침

### 보안 테스트

- 허용되지 않은 origin 차단
- API 토큰 없는 요청 차단
- 과도한 steps 요청 차단
- 비정상 확장자 업로드 차단
- 프롬프트 및 파일 크기 제한

## 16. 비용과 운영 고려사항

Vercel은 프론트엔드 요청을 처리하지만 GPU 비용은 Elice 또는 별도 GPU 서비스에서 발생한다. 특히 인스턴스가 계속 실행되면 사용하지 않는 시간에도 비용이 발생할 수 있다.

권장 운영 방식:

- 개발 시에만 GPU 인스턴스 실행
- 작업 큐가 비었을 때 인스턴스 중지 검토
- 결과 이미지 보존 기간 설정
- 생성 steps와 해상도 제한
- 사용자별 요청량 제한

## 17. 최종 권장안

### 개발·실습용

```text
Elice A100 MIG
  ├── FastAPI 또는 Gradio
  └── SSH tunnel

로컬 브라우저
```

### 공개 웹서비스용

```text
Vercel Next.js
  ↓ HTTPS + API Token
공개 HTTPS FastAPI
  ↓
GPU worker + Redis
  ↓
A100 MIG
  ↓
S3 + PostgreSQL
```

Vercel과 GPU 서버를 분리하면 각 플랫폼의 역할이 명확해지고, 프론트엔드 배포와 모델 추론을 독립적으로 운영할 수 있다.

## 18. 결론

현재 Gradio는 SDXL 모델의 기능을 빠르게 검증하는 데 적합한 프로토타입이다. 하지만 공개 웹서비스로 확장하려면 Vercel에 GPU 추론 코드를 직접 올리는 것이 아니라, Vercel을 프론트엔드로 사용하고 A100 MIG 서버를 인증된 HTTPS 추론 백엔드로 운영해야 한다.

가장 안전한 진행 순서는 다음과 같다.

1. 현재 CLI·Gradio 생성 품질과 메타데이터 안정화
2. FastAPI 비동기 생성 API 구현
3. 단일 GPU 작업 큐와 상태 조회 구현
4. Next.js 기반 UI 구현
5. Vercel Preview 배포
6. HTTPS·인증·CORS 검증
7. Production 배포
8. 저장소·모니터링·요청 제한 추가
