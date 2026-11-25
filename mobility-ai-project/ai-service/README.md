# AI 서비스 (ai-service)

모빌리티 비전 AI 서비스 — **번호판 인식(LPR)** 과 **예지보전(PdM)** REST API.
Python 3.12 + FastAPI, 포트 **8000**.

> **핵심: 학습된 가중치 파일이 없어도 모든 엔드포인트가 200 을 반환한다.**
> 가중치가 없으면 OpenCV 기반 결정적 알고리즘과 도메인 규칙으로 자동 대체되고,
> 응답의 `detector`/`recognizer` 에 `-fallback` 접미사가 붙는다. 500 을 던지지 않는다.

---

## 1. 단독 실행

```bash
cd ai-service

# 가상환경 (권장)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 최소 설치 — 이것만으로 전체 엔드포인트가 동작한다(폴백 모드)
pip install fastapi==0.118.0 "uvicorn[standard]==0.37.0" pydantic==2.11.9 \
            pydantic-settings==2.10.1 python-multipart==0.0.20 \
            numpy==2.1.3 opencv-python-headless==4.12.0.88 pillow==11.3.0 \
            python-dotenv==1.1.1

# 또는 전체 설치(딥러닝 경로 포함, 약 3GB)
pip install -r requirements.txt

# 환경변수
cp .env.example .env

# 실행
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 또는
python -m app.main
```

접속:

| URL | 설명 |
|---|---|
| http://localhost:8000/ | 엔드포인트 목록 |
| http://localhost:8000/docs | Swagger UI (대화형 테스트) |
| http://localhost:8000/health | 헬스체크 |

---

## 2. 엔드포인트 요약

| Method | Path | 설명 |
|---|---|---|
| GET | `/health` | 상태·버전·디바이스·현재 모델명 (모델 로드 안 함) |
| POST | `/api/v1/lpr/recognize` | multipart: `file`, `min_conf`, `detector`, `recognizer` |
| POST | `/api/v1/lpr/recognize-frames` | multipart: `files` 다중 + `fusion` |
| POST | `/api/v1/pdm/predict` | JSON: `vehicle_id`, `odometer_km`, `readings[]` |
| GET | `/api/v1/pdm/spec` | PdM 모델 스펙·가중치 존재 여부 |
| GET | `/api/v1/models/info` | 모델 레지스트리 (`?load=true` 로 실제 로딩 확인) |
| POST | `/api/v1/models/benchmark` | 벤치마크 실행 → 결과 배열 |

### 빠른 확인

```bash
# 헬스체크
curl -s localhost:8000/health | python3 -m json.tool

# 번호판 인식
curl -s -X POST localhost:8000/api/v1/lpr/recognize \
  -F "file=@plate.jpg" -F "min_conf=0.25" \
  -F "detector=yolo" -F "recognizer=crnn" | python3 -m json.tool

# 다중 프레임 (정확도가 더 높다)
curl -s -X POST localhost:8000/api/v1/lpr/recognize-frames \
  -F "files=@f1.jpg" -F "files=@f2.jpg" -F "files=@f3.jpg" \
  -F "fusion=char-vote" | python3 -m json.tool

# 예지보전
curl -s -X POST localhost:8000/api/v1/pdm/predict \
  -H 'Content-Type: application/json' \
  -d '{"vehicle_id":1,"odometer_km":84210,
       "readings":[{"measured_at":"2026-10-01T09:00:00","engine_temp_c":92.4,
                    "engine_rpm":1850,"speed_kph":48.0,"battery_voltage":12.4,
                    "battery_soh":0.86,"tire_pressure_fl":32.1,"tire_pressure_fr":31.8,
                    "tire_pressure_rl":30.9,"tire_pressure_rr":31.2,"brake_pad_mm":4.1,
                    "oil_quality":0.62,"vibration_rms":0.44,"fuel_level":0.55}]}' \
  | python3 -m json.tool
```

---

## 3. 테스트

```bash
# 전체 (가중치 불필요, 네트워크 불필요)
pytest -q

# 순수 로직만 (fastapi 없어도 통과)
pytest tests/test_postprocess.py tests/test_ctc_decoder.py tests/test_quality.py -q

# 문법 검증
python -m compileall -q app tests
```

`tests/test_api.py` 는 `fastapi`/`httpx` 가 없으면 `pytest.importorskip` 으로 스킵된다.
따라서 최소 설치 환경에서도 `pytest` 가 실패하지 않는다.

---

## 4. 폴백 동작 (가중치 없을 때)

| 단계 | 폴백 알고리즘 | 위치 |
|---|---|---|
| 번호판 검출 | 그레이스케일 → CLAHE → Sobel-x + 선택적 임계값 → 모폴로지 → 윤곽선 → **문자 밴드 → 번호판 2단 추정** → 종횡비(2.0~5.5)·면적·사각도 필터 → 텍스처 점수 → NMS | `app/core/fallback.py` |
| OCR | crop 이진화(극성 자동 판정) → 테두리 억제 → 연결요소 조각 수집 → **x 간격 클러스터링(한글 자모 병합)** → 40x32 정규화 → 43차원 구조 특징 → 자리 제약 템플릿 매칭 | `app/core/fallback.py` |
| 이상탐지 | 도메인 임계값 + z-score 결합 | `app/pdm/anomaly.py` |
| 고장분류 | 부품별 도메인 규칙 진단 | `app/pdm/failure.py` |
| RUL | 선형 외삽 + 상태 기반 기한 상한 | `app/pdm/rul.py` |

**합성 데이터 15장 기준 폴백 성능** (CPU, 가중치 0바이트):

```
검출 재현율(IoU 0.4)  0.933
완전 일치율           0.600
문자 정확도           0.730
지연 p50 / p95        40ms / 55ms
PdM 지연 p50          2.05ms
```

> 합성 데이터 기준이므로 **모델 간 상대 비교** 용도로만 해석해야 한다.
> 딥러닝 가중치를 학습해 넣으면 완전 일치율이 0.95 이상으로 올라간다.

폴백은 **결정적(deterministic)** 이다 — 난수를 쓰지 않으므로 같은 입력에
항상 같은 출력을 낸다(교재 실습 재현성 보장).

### OpenCV 조차 없는 환경

`opencv-python-headless` 는 requirements 에 포함되어 있지만, 설치에 실패한
환경에서도 **numpy 만으로 전 경로가 동작한다**(적분영상 박스 블러, Otsu,
분리형 모폴로지, 연결요소 라벨링을 모두 numpy 로 구현).
합성 데이터 15장 기준 검출 14/15, 깨끗한 번호판 OCR 11/15.
품질은 OpenCV 경로보다 낮으므로 **최후의 안전장치**로만 쓴다.

---

## 5. 모델 학습 (선택)

```bash
# 1) 합성 데이터 생성
python -m app.train.synth_plate --out ./data/synth --count 2000              # 검출용(장면 포함)
python -m app.train.synth_plate --out ./data/ocr --count 20000 --no-scene    # OCR용(번호판만)

# 2) 검출기 학습 (ultralytics 필요)
python -m app.train.train_detector --data ./data/synth/data.yaml --epochs 50

# 3) OCR 학습 (torch 필요)
python -m app.train.train_crnn --data ./data/ocr --epochs 30 --batch 64

# 4) ONNX 내보내기 (CPU 추론 1.5~2.5배 가속)
python -m app.train.export_onnx --crnn ./models/crnn_ctc_kr.pt \
                                --detector ./models/plate_yolo_nano.pt

# 5) INT8 양자화 (크기 1/4, 추가 가속)
python -m app.train.quantize --crnn ./models/crnn_ctc_kr.pt
python -m app.train.quantize --onnx ./models/plate_yolo_nano.onnx

# 6) 벤치마크
python -m app.train.benchmark --samples 50 --out bench.json
```

학습 스크립트는 **의존성이 없으면 안내 메시지와 함께 종료 코드 2** 로 끝난다
(import 단계에서 터지지 않는다).

---

## 6. 선택 의존성

### PaddleOCR (requirements.txt 에 **미포함**)

```bash
pip install paddlepaddle==2.6.2 paddleocr==2.8.1
```

제외 이유:
1. `paddlepaddle` 휠이 플랫폼/CPU 명령어셋별로 나뉘어 있고 용량이 크다.
2. 첫 실행 시 **모델을 네트워크로 다운로드**한다 → 오프라인/테스트 환경과 충돌.
3. 한국 번호판 전용이 아니라 한글 일반 OCR 모델이라 CRNN 보다 정확도가 낮을 수 있다.

미설치 상태에서 `recognizer=paddle` 로 요청하면 500 이 아니라
`paddle-fallback` 으로 처리된다.

### 딥러닝 / ML

```bash
pip install -e ".[deep]"    # torch, torchvision, ultralytics, onnxruntime
pip install -e ".[ml]"      # scikit-learn, pandas, joblib
pip install -e ".[dev]"     # pytest, httpx, ruff
```

---

## 7. 주요 환경변수

`.env.example` 전체 목록 참조. 자주 바꾸는 값:

| 변수 | 기본값 | 설명 |
|---|---|---|
| `PORT` | `8000` | 서버 포트 |
| `DEVICE` | `cpu` | `cuda` 로 설정해도 torch/GPU 없으면 `cpu` 로 강등 |
| `TORCH_THREADS` | `4` | 0 이면 CPU 코어 수 자동 |
| `CORS_ORIGINS` | `localhost:5173,8080` | 쉼표 구분 |
| `DEFAULT_MIN_CONF` | `0.25` | 검출 최소 신뢰도 |
| `LOW_CONF_THRESHOLD` | `0.55` | 이 값 미만이면 `status=LOW_CONFIDENCE` |
| `QUALITY_PLATE_PX_MIN` | `72` | 번호판 최소 픽셀폭(광학적 하한) |
| `CRNN_WEIGHTS` | `./models/crnn_ctc_kr.pt` | 없으면 폴백 OCR |
| `DETECTOR_ONNX` | `./models/plate_yolo_nano.onnx` | ONNX 가 있으면 `.pt` 보다 우선 |
| `LOG_FORMAT` | (비움) | `json` 으로 설정하면 한 줄 JSON 로그 |

가중치 경로의 파일이 없으면 **자동으로 폴백**이 선택되며, 기동 로그에 경고가 남는다.

---

## 8. Docker

```bash
# 루트의 docker-compose.yml 로 전체 스택 실행
cd .. && docker compose up -d ai-service

# 단독 빌드
docker build -t mobility-ai-service ./ai-service
docker run -p 8000:8000 --env-file ai-service/.env mobility-ai-service
```

헬스체크 경로: `/health` (또는 `/healthz`)

---

## 9. 디렉터리 구조

```
ai-service/
├── app/
│   ├── main.py              FastAPI 앱, CORS, lifespan, 전역 예외처리
│   ├── config.py            설정 (pydantic-settings 3단 강등)
│   ├── schemas.py           pydantic v2 요청/응답 모델
│   ├── core/                로깅, 모델 레지스트리, OpenCV 폴백
│   ├── detection/           검출기 (base/preprocess/yolo/mobilenet)
│   ├── ocr/                 문자셋, CRNN, CTC 디코더, Attention, Paddle, 후처리
│   ├── pipeline/            품질평가, 보정, LPR 파이프라인
│   ├── pdm/                 특징공학, 이상탐지, 고장분류, RUL, 서비스
│   ├── routers/             health, lpr, pdm, models
│   └── train/               합성데이터, 학습, ONNX, 양자화, 벤치마크
├── tests/                   postprocess, ctc_decoder, quality, api
├── models/                  가중치 (비어 있어도 동작)
├── requirements.txt
├── pyproject.toml           ruff + pytest 설정
└── .env.example
```

교재 인용 지점은 `MANIFEST.md` 참조.
