# 실험 및 버전 관리

## 파일명 규칙

`YYYYMMDD_<type>_<experiment-id>_<short-name>.<ext>`

예: `20250115_t2i_exp0001_korean-teahouse.png`

## 메타데이터 필수 항목

모델 ID와 버전, prompt, negative prompt, seed, width, height, steps, guidance scale, dtype, LoRA 경로·weight, 생성 시간, Git commit을 기록한다.

## Git 원칙

- 코드·문서·설정만 커밋한다.
- 모델 가중치, 개인 데이터, API 토큰, 대용량 결과물은 `.gitignore`로 제외한다.
- 의미 있는 단위로 커밋한다. 예: `docs: add inference guide`, `feat: add txt2img service`
- 모델과 데이터의 출처·라이선스는 별도 문서에 기록한다.
