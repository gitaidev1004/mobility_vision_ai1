# AI 서비스 MANIFEST — 파일 목록 / 검증 결과 / 교재 인용 지점

생성일: 2026-10-24
대상: `mobility-ai-project/ai-service` (계약서 3장 "AI 서비스 REST 계약")
총 규모: **47개 파일(.py 43 + 메타 4), 18,723 라인** (app 16,782 / tests 1,941)

---

## 1. 파일 목록 (1줄 설명)

### 1.1 프로젝트 메타

| 파일 | 라인 | 설명 |
|---|---|---|
| `requirements.txt` | 18 | Python 3.12 기준 핀 고정 의존성 (PaddleOCR 제외) |
| `pyproject.toml` | 134 | `[project]` 메타 + optional-dependencies(deep/ml/paddle/dev) + ruff + pytest 설정 |
| `.env.example` | 70 | 전체 환경변수 예시와 각 값의 선택 근거 주석 |
| `README.md` | 240 | 단독 실행법, 엔드포인트, 폴백 성능 수치, 학습 파이프라인 |
| `MANIFEST.md` | — | 이 문서 |

### 1.2 애플리케이션 코어

| 파일 | 라인 | 설명 |
|---|---|---|
| `app/__init__.py` | 17 | 패키지 버전(`1.0.0`)과 3대 설계 원칙 선언 |
| `app/config.py` | 419 | pydantic-settings → pydantic(`create_model`) → 순수객체 **3단 강등** 설정 + `AI_` 접두사 지원 |
| `app/schemas.py` | 422 | 계약서 JSON 과 1:1 대응하는 pydantic v2 요청/응답 모델 |
| `app/main.py` | 539 | FastAPI 앱, CORS, lifespan, 요청ID 미들웨어, **전역 예외 → 계약 응답 변환** |
| `app/core/logging.py` | 185 | ContextVar 요청ID 전파 + JSON/텍스트 포매터 (stderr 출력) |
| `app/core/registry.py` | 504 | 모델 레지스트리: 이름→로더, 지연 로딩, 이름별 락 싱글턴 |
| `app/core/fallback.py` | **1,608** | **가중치 0바이트 결정적 추론** — OpenCV/numpy 검출 + 템플릿 OCR |

### 1.3 검출 (detection)

| 파일 | 라인 | 설명 |
|---|---|---|
| `app/detection/base.py` | 216 | `PlateDetector` ABC + `Detection`/`DetectorInfo` 데이터 구조 |
| `app/detection/preprocess.py` | 433 | letterbox, 역변환, NMS, IoU, 이미지 디코딩 (numpy 전용) |
| `app/detection/yolo_detector.py` | 402 | ultralytics(.pt) + ONNXRuntime(.onnx) 양쪽 경로 래퍼 |
| `app/detection/mobilenet_detector.py` | 330 | torchvision SSDLite320-MobileNetV3 전이학습 검출기 |

### 1.4 OCR

| 파일 | 라인 | 설명 |
|---|---|---|
| `app/ocr/charset.py` | 334 | 한국 번호판 문자셋 — 숫자 10 + 용도문자 51 + 지역명 17 → 78 클래스 |
| `app/ocr/crnn.py` | 338 | PyTorch CRNN (CNN 7층 + BiLSTM 2층 + CTC) 모델 정의 |
| `app/ocr/ctc_decoder.py` | 504 | **순수 numpy** CTC greedy + prefix beam search 디코더 |
| `app/ocr/attention_ocr.py` | 439 | CNN + Bahdanau Attention Seq2Seq OCR 모델 정의 |
| `app/ocr/paddle_adapter.py` | 242 | PaddleOCR 선택적 어댑터 (2.x/3.x 출력 형식 모두 파싱) |
| `app/ocr/recognizer.py` | 600 | 인식기 팩토리 — 실패를 폴백으로 흡수(예외 미전파) |
| `app/ocr/postprocess.py` | 744 | 정규식 검증, **자리 기반 혼동 보정**, 다중 프레임 융합 |

### 1.5 파이프라인

| 파일 | 라인 | 설명 |
|---|---|---|
| `app/pipeline/quality.py` | 491 | 라플라시안 분산/밝기/대비/포화/픽셀폭 품질 평가 |
| `app/pipeline/rectify.py` | 576 | Hough 기울기 추정, 투시 변환, CLAHE, 언샤프 마스킹 |
| `app/pipeline/lpr_pipeline.py` | 504 | 검출→보정→OCR→후처리 + **후보 경쟁(candidate competition)** |

### 1.6 예지보전 (PdM)

| 파일 | 라인 | 설명 |
|---|---|---|
| `app/pdm/features.py` | 553 | 전방채움 보간, 이상값 클리핑, rolling 통계, 윈도우, 56차원 특징 |
| `app/pdm/anomaly.py` | 613 | IsolationForest/OneClassSVM/AutoEncoder + **통계 임계 폴백** |
| `app/pdm/failure.py` | 741 | RandomForest/GradientBoosting + **부품별 도메인 규칙 진단** |
| `app/pdm/rul.py` | 470 | LSTM RUL + 선형 외삽 폴백 + 샘플 간격 자동 추정 |
| `app/pdm/service.py` | 583 | 3모델 앙상블 → `health_score`/`risk_level`/권고 생성 |

### 1.7 라우터

| 파일 | 라인 | 설명 |
|---|---|---|
| `app/routers/health.py` | 134 | `GET /health`, `/healthz` — 모델 로드 없이 상태 보고 |
| `app/routers/lpr.py` | 401 | `POST /lpr/recognize`, `/recognize-frames` + 파라미터 정규화 |
| `app/routers/pdm.py` | 145 | `POST /pdm/predict`, `GET /pdm/spec` |
| `app/routers/models.py` | 213 | `GET /models/info`, `POST /models/benchmark` |

### 1.8 학습 / 운영 CLI

| 파일 | 라인 | 설명 |
|---|---|---|
| `app/train/synth_plate.py` | 794 | 합성 번호판 생성기 — 4종 규격 + 투시/블러/조명/노이즈/JPEG 증강 |
| `app/train/train_detector.py` | 304 | YOLO 학습 CLI (번호판 특화 증강 설정 + EarlyStopping) |
| `app/train/train_crnn.py` | 599 | CRNN+CTC 학습 루프 (체크포인트/EarlyStopping/기울기 클리핑) |
| `app/train/export_onnx.py` | 360 | ONNX 내보내기 + **PyTorch 대조 수치 검증** |
| `app/train/quantize.py` | 373 | INT8 동적 양자화 (PyTorch LSTM/Linear + ONNX Conv) |
| `app/train/benchmark.py` | 559 | CPU 지연(p50/p95)/메모리(RSS 증분)/정확도 벤치마크 |

### 1.9 테스트

| 파일 | 라인 | 테스트 수 | 설명 |
|---|---|---|---|
| `tests/test_postprocess.py` | 465 | 72 | 정규식, 혼동 보정, 상태 판정, 프레임 융합 (numpy 불필요) |
| `tests/test_ctc_decoder.py` | 332 | 32 | CTC 축약 규칙, greedy/beam 디코딩, 수치 안정성 |
| `tests/test_quality.py` | 381 | 28 | 블러/밝기/포화/픽셀폭 평가 + 계약 스키마 검증 |
| `tests/test_api.py` | 763 | 40+ | TestClient 로 전 엔드포인트 200 확인 (`importorskip` 스킵 가능) |

---

## 2. 검증 결과

### 2.1 문법 검증

```
$ python3 -m compileall -q app tests
→ 통과 (46개 파일, 오류 없음)
```

### 2.2 pytest (네트워크 없이, 가중치 없이)

```
$ python3 -m pytest tests -q
........................................................................ [ 54%]
............................................................             [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_api.py:20: FastAPI 미설치 — API 테스트 스킵
132 passed, 1 skipped in 0.28s
```

- **순수 로직 테스트 132건 전부 실제 실행 통과** (postprocess / ctc_decoder / quality)
- `test_api.py` 는 검증 환경에 fastapi/httpx 가 없어 `pytest.importorskip` 으로 스킵
  (fastapi 설치 환경에서는 40여 건이 추가 실행된다)
- 사용 라이브러리: numpy 2.5.3, opencv 5.0.0, pillow 12.3.0, pytest 9.1.1

### 2.3 품질 지표

| 항목 | 결과 |
|---|---|
| 공개 함수/클래스 docstring (한국어) | **100%** (누락 0건) |
| 함수 타입힌트 (인자 + 반환) | **100%** (누락 0건) |
| 모듈 docstring | 46/46 |

### 2.4 폴백 성능 벤치마크 (가중치 0바이트, CPU)

```
$ python3 -m app.train.benchmark --samples 15

model_name                                task        recall  exact   p50      p95      char_acc
pdm-ensemble-1.0.0                        pdm         0.000   0.000    2.05ms   2.22ms   —
mobilenet-fallback + attention-fallback   detect+ocr  0.933   0.600   34.72ms  39.97ms  0.730
mobilenet-fallback + crnn-fallback        detect+ocr  0.933   0.600   37.44ms  46.49ms  0.730
yolo-fallback + attention-fallback        detect+ocr  0.933   0.600   38.46ms  45.59ms  0.730
yolo-fallback + crnn-fallback             detect+ocr  0.933   0.600   40.17ms  55.03ms  0.730
```

깨끗한 합성 번호판 단독 crop 기준 OCR 완전 일치: **6/6**
증강 없는 장면 이미지 30장 기준: 검출 30/30, 완전 일치 20/30 (67%)

### 2.5 계약서 준수 확인 (핸들러 직접 호출)

| 입력 | 결과 |
|---|---|
| 번호판 crop PNG | `SUCCESS` / `12가3456` / 계약 필드 11개 전부 존재 |
| 장면 이미지 PNG | `SUCCESS` / 계약 필드 전부 존재 |
| 이미지 아닌 바이트 | `NO_PLATE` + `quality.reason` 설정 (**500 아님**) |
| 빈 파일 | `NO_PLATE` (**500 아님**) |
| 알 수 없는 모델명 | 기본값으로 처리 후 200 (**422 아님**) |
| `detector=mobilenet` | `mobilenet-fallback` |
| `recognizer=paddle` (미설치) | `paddle-fallback` (**500 아님**) |
| PdM 빈 readings | `MEDIUM` / health 60 (**"건강함" 으로 오보고 안 함**) |
| PdM 비정상 값(-999 등) | 200 + 계약 스키마 |
| 벤치마크 | 계약서 9개 필드 포함 배열 |

### 2.6 개발 중 실측으로 발견·수정한 결함 (교재 "흔한 함정" 소재)

| # | 증상 | 원인 | 수정 |
|---|---|---|---|
| 1 | 폴백 검출 재현율 0/25 | Otsu 가 단봉 노이즈 분포를 갈라 전경 비율 0.993 → 화면 전체가 한 덩어리 | `_selective_threshold` — Otsu 와 분위수 임계값 중 **큰 값** 사용 |
| 2 | 밴드는 찾는데 검출 0건 | 문자 밴드 종횡비(8~9)에 번호판 필터(2.0~5.5) 적용 | 2단 설계: 밴드(1.5~14) → 번호판 박스 가설 확장 → 최종 필터 |
| 3 | OCR 전면 오인식 (`12가3456`→`227호3456`) | Pillow 기본 비트맵 폰트(6x11)를 확대해 획 구조 소실 | TrueType 폰트 supersampling 렌더링 |
| 4 | 7자 번호판이 8개로 분할 | 테두리선이 전경 성분으로 남음 | `_suppress_border` — 테두리 띠를 가로지르는 거대 성분 제거 |
| 5 | 한글 자모 분리 (`가`=ㄱ+ㅏ) | "연결요소 = 문자" 가정이 한글에서 성립하지 않음 | x 간격 클러스터링 (중앙폭 x 0.42 임계값) |
| 6 | 연결요소 0개 (`segmentation-failed`) | "전경 과반이면 반전" 규칙이 차체 섞인 crop 에서 오작동 | 중앙 60% 영역의 잉크 비율(0.20)로 극성 판정 |
| 7 | 올바른 글자 8개 중 7개 탈락 | `h >= plate_h * 0.30` — crop/번호판 비율 가정이 깨짐 | 높이 **중앙값 대비** 0.55~1.70배 self-referential 필터 |
| 8 | 기울기 추정이 항상 0 | 번호판 테두리선(양 끝이 경계에 닿음)을 전부 제외 | "같은 변을 따라 달리는" 선분만 제외 |
| 9 | `HoughLinesP` TypeError | OpenCV 5.x 는 `(N,4)`, 4.x 는 `(N,1,4)` 반환 | `_normalize_hough_output` 로 양쪽 흡수 |
| 10 | 선명한 흰 번호판을 "노출 과다" 반려 | 흰 바탕 crop 평균이 215 로 정상적으로 상한 초과 | 포화 비율(>10%) 또는 저대비 증거 동반 시에만 반려 |
| 11 | 위험 고정 차량 < 악화 중 차량 (0.571 < 0.657) | z-score 가 0 이라 가중합이 낮아짐 | z-score 는 심각도를 **올리기만** 하도록 `max()` 결합 |
| 12 | RUL 51일인데 "이상 징후 없음" 권고 | 상태(NORMAL)만 보고 RUL 무시 | RUL < 90일도 권고 대상에 포함 |
| 13 | CRITICAL 배터리 "730일 뒤 점검" | 값이 일정해 기울기 0 → RUL 상한 | 상태 기반 기한 상한 (CRITICAL 3일 / WARNING 14일) |
| 14 | 빈 readings → health 88.8 / LOW | 중립값들이 높은 점수로 합성 | `frame.rows == 0` 특별 처리 → MEDIUM 고정 |
| 15 | beam search 쓰레기 후보 신뢰도 1.0 | 도달 불가 접두사(`log_prob = -1e30`)가 후보에 포함 | 도달 가능 후보만 반환 + 길이 정규화 |
| 16 | 1프레임과 3프레임 합의 보너스 동일 | 비율만 봐서 1/1 과 3/3 이 같음 | 독립 관측 수 `log(votes)` 보너스 추가 |
| 17 | 벤치마크 JSON 파이프 파싱 실패 | 로그가 stdout 에 섞임 | 로그를 stderr 로 (데이터=stdout, 진단=stderr) |
| 18 | 번호판 crop 입력 시 검출 0건 | 세로 확장이 경계에 막혀 종횡비 탈락 | "이미지 전체가 번호판" 가설 추가 |
| 19 | 조기 반환 경로가 `-fallback` 누락 | 정상 경로와 다른 이름 보고 → 집계 불일치 | `exposed_model_names` 로 이름 결정 통합 |
| 20 | `compose_scene` 브로드캐스트 오류 | `(H,1,3)` 배열에 in-place 연산 | 명시적 `(H,W,3)` 실배열 확장 |
| 21 | **numpy-only 경로 검출 1/10** | OpenCV 경로의 `GaussianBlur` 에 대응하는 평활화가 numpy 경로에 없어 노이즈가 그대로 미분됨 | 적분영상 박스 블러 추가 + 커널 25 → **14/15** |
| 22 | `docker-compose.yml` 의 `AI_TORCH_THREADS=4` 가 무시됨 | 인프라는 `AI_` 접두사, 설정은 접두사 없는 이름을 읽음 | `ENV_PREFIXES` 로 두 형식 모두 지원(`AI_` 우선) |
| 23 | **pydantic 폴백에서 32개 필드 중 4개만 등록** | 클래스 본문 뒤 `__annotations__` 주입 + `model_rebuild` 는 v2 에서 동작하지 않음(메타클래스가 생성 시점에 필드 수집) | `create_model` 로 클래스를 처음부터 생성 |

---

## 3. 교재 인용 지점 (장별 매핑)

> 각 항목은 `파일:함수` — **왜 이 코드가 교재에 적합한가** 형식이다.

### 3장 — 서비스 아키텍처 / 의존성 설계

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/config.py:Settings` (3단 강등 블록) | 의존성이 없을 때도 import 가 깨지지 않는 설정 계층 설계 |
| `app/config.py` `create_model` 블록 | **pydantic v2 에서 동적 필드를 만드는 올바른 방법** — 결함 #23 |
| `app/config.py:ENV_PREFIXES` / `_read_env` | 인프라(`AI_PORT`)와 문서(`PORT`) 의 이름 규약 불일치 흡수 — 결함 #22 |
| `app/config.py:get_settings` | `lru_cache` 싱글턴 + `cache_clear` 로 테스트 가능성 확보 |
| `app/config.py:Settings.resolved_device` | `DEVICE=cuda` 여도 GPU 가 없으면 `cpu` 로 조용히 강등 |
| `app/main.py:create_app` | CORS 와 자격증명의 CORS 규격 충돌(`allow_origins=["*"]`) 분기 |
| `app/main.py:_apply_thread_limits` | BLAS/OpenMP 스레드 제한이 **처리량을 올리는** 이유 (thrashing) |
| `app/main.py:lifespan` | `on_event` 대신 lifespan, **기동 시 모델 로드 금지**(지연 로딩) |
| `app/core/registry.py:ModelRegistry.get` | double-checked locking 으로 중복 모델 로딩(OOM 위험) 방지 |
| `app/core/registry.py:make_recognizer_loader` | 루프 변수 캡처 버그를 클로저 팩토리로 회피 |
| `app/core/logging.py:configure_logging` | 데이터=stdout / 진단=stderr 규약, 멱등 설정 |
| `app/core/logging.py` ContextVar 블록 | 비동기 환경에서 전역 변수가 요청 간 값을 섞는 문제 |

### 4장 — 번호판 검출 (전통 영상처리)

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/core/fallback.py:_clahe` | 전역 `equalizeHist` 가 역광에서 실패하는 이유, CLAHE 의 clipLimit |
| `app/core/fallback.py:_edge_map` | **x 방향 Sobel 만 쓰는 이유** — 번호판의 수직획 반복 텍스처 |
| `app/core/fallback.py:_selective_threshold` | **Otsu 의 실패 조건** (단봉 분포) 과 분위수 하이브리드 — 결함 #1 |
| `app/core/fallback.py:_otsu_threshold` | Otsu 를 numpy 로 유도 (클래스 간 분산 최대화) |
| `app/core/fallback.py:_numpy_box_blur` | 적분영상으로 O(H·W) 박스 블러 — 포함-배제 원리 |
| `app/core/fallback.py:_edge_map` (numpy 경로) | **전처리 단계 누락이 성능을 10배 떨어뜨린 사례** — 결함 #21 |
| `app/core/fallback.py:_slide_extreme` | 분리형 슬라이딩 최대/최소로 dilate/erode 유도 |
| `app/core/fallback.py:_candidate_bands_from_edges` | 밴드 vs 번호판 종횡비 구분 — 결함 #2 |
| `app/core/fallback.py:_bands_to_plate_boxes` | **generate-and-test** 패턴 (가설 생성은 싸게, 선택은 점수로) |
| `app/core/fallback.py:_plate_texture_score` | 기하 조건만으로 창문/스티커를 못 거르는 이유, 3신호 결합 |
| `app/core/fallback.py:_normalize_angle` | `minAreaRect` 각도의 버전별 모호성 해소 |
| `app/detection/preprocess.py:letterbox` | 종횡비 보존이 학습/추론 도메인 차이를 없애는 원리 |
| `app/detection/preprocess.py:unletterbox_boxes` | **역변환 순서**(패딩 먼저, 배율 나중)가 중요한 이유 |
| `app/detection/preprocess.py:nms` | NMS 를 numpy 로 유도 + 브로드캐스팅 IoU 행렬 |
| `app/detection/preprocess.py:resize` | 축소=INTER_AREA / 확대=INTER_LINEAR 선택이 OCR 정확도에 미치는 영향 |

### 5장 — 딥러닝 검출기 (YOLO / MobileNet)

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/detection/base.py:PlateDetector` | 의존성 역전 — 모델 교체가 파이프라인을 건드리지 않는 인터페이스 |
| `app/detection/base.py:Detection.expand` | 검출 박스 3~8% 확장이 OCR 이진화를 안정시키는 이유 |
| `app/detection/yolo_detector.py:_decode_onnx_output` | **YOLO v5/v8/NMS포함 출력 레이아웃 차이**를 흡수하는 디코딩 |
| `app/detection/yolo_detector.py:_init_onnx` | ONNXRuntime 세션 옵션(스레드/그래프 최적화) 의 실측 효과 |
| `app/detection/yolo_detector.py:YoloPlateDetector.__init__` | ONNX 우선 선택 근거 (런타임 2.5GB → 60MB) |
| `app/detection/mobilenet_detector.py:build_ssdlite_detector` | **전이학습 표준 패턴** — 백본 유지 + 분류 헤드만 교체 |
| `app/detection/mobilenet_detector.py:_detect_impl` | torchvision 검출 모델의 입력 규약(RGB/0~1/내부 리사이즈) |
| `app/detection/mobilenet_detector.py` 모듈 docstring | YOLO vs SSDLite 비교표 (속도/라이선스/작은 객체) |

### 6장 — OCR 모델 (CRNN / CTC / Attention)

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/ocr/charset.py` 모듈 docstring + `CHARSET` | 한글 11,172자를 51자로 줄이는 것이 정확도에 결정적인 이유 |
| `app/ocr/charset.py:nearest_use_char` | 유니코드 한글 초성/중성 분해로 **실제 혼동 패턴**에 맞는 보정 |
| `app/ocr/crnn.py:build_crnn` (CNN 풀링 블록) | **비대칭 풀링**(높이 32→1, 너비 160→40)이 CTC 의 핵심인 이유 |
| `app/ocr/crnn.py:CRNN.log_probs_for_ctc` | `(B,T,C)` → `(T,B,C)` permute 를 모델에 캡슐화해 실수 방지 |
| `app/ocr/ctc_decoder.py:collapse` | **중복 제거 → blank 제거 순서**가 바뀌면 `aa` 를 표현 못 함 |
| `app/ocr/ctc_decoder.py:beam_search_decode` | prefix beam search 의 `p_blank`/`p_nonblank` 분리 이유 |
| `app/ocr/ctc_decoder.py:softmax` | 최댓값 차감 없이는 `exp(1000)` 이 inf 가 되는 수치 안정성 |
| `app/ocr/ctc_decoder.py:_geometric_mean` | 번호판에서 산술평균이 아니라 기하평균을 쓰는 운영적 근거 |
| `app/ocr/attention_ocr.py` 모듈 docstring | **CTC vs Attention 비교표** (단조 정렬/조건부 독립/추론 비용) |
| `app/ocr/attention_ocr.py:BahdanauAttention` | additive vs dot-product attention 선택 근거 |
| `app/ocr/attention_ocr.py:AttentionOCR.forward` | **teacher forcing** 과 exposure bias 트레이드오프 |
| `app/ocr/attention_ocr.py:AttentionOCR.greedy_decode` | EOS 조기 종료와 완료 플래그 처리 |

### 7장 — 폴백 OCR (학습 없는 문자 인식)

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/core/fallback.py:_binarize_plate` | 흰/녹/노 바탕 번호판의 **극성 통일** 필요성 |
| `app/core/fallback.py:_choose_polarity` | 전역 과반 규칙의 실패와 **중앙 영역 잉크 비율** 판정 — 결함 #6 |
| `app/core/fallback.py:_suppress_border` | 테두리선이 거대 연결요소가 되는 문제 — 결함 #4 |
| `app/core/fallback.py:_segment_characters` | **"연결요소 = 문자" 가정이 한글에서 깨지는 이유** — 결함 #5 |
| `app/core/fallback.py:_cluster_fragments_into_glyphs` | 자간 vs 자모 간격을 중앙폭 비례 임계값으로 가르기 |
| `app/core/fallback.py:_filter_glyphs` | **self-referential filtering** — 외부 기준 대신 글자끼리 비교 — 결함 #7 |
| `app/core/fallback.py:_glyph_features` | 43차원 수작업 특징 설계 (영역/투영/위상) 와 격자 수 실측 근거 |
| `app/core/fallback.py:_count_holes` | 오일러 특성(구멍 수)으로 `0`/`8`/`9` 식별 |
| `app/core/fallback.py:_match_glyph` | 코사인 + L1 결합 이유, **폴백 신뢰도 상한 0.75** 의 운영적 근거 |
| `app/core/fallback.py:_render_template` | supersampling 렌더링 — 결함 #3 |
| `app/core/fallback.py:_layout_for` | **자리 제약**이 탐색 공간을 10배 줄이는 효과 |

### 8장 — 후처리 (도메인 규칙의 힘)

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/ocr/postprocess.py:correct_confusions` | **자리 기반 보정** vs 전역 치환 — 왜 맥락이 필수인가 |
| `app/ocr/postprocess.py:USE_CHAR_CONFUSION` 주석 | **`0 → O` 역보정 금지** 규칙의 논리적 근거 |
| `app/ocr/postprocess.py:is_valid_plate` | 정규식 `[가-힣]` 만으로 부족하고 화이트리스트가 필요한 이유 |
| `app/ocr/postprocess.py:split_region` | 3번째 글자가 숫자인지로 지역명을 판정하는 구조적 근거 |
| `app/ocr/postprocess.py:decide_status` | **형식 검증이 신뢰도보다 먼저**인 운영적 이유 |
| `app/ocr/postprocess.py:postprocess` | 보정 벌점 — "쓰레기를 고쳐 0.9 로 보고" 를 막는 정직성 |
| `app/ocr/postprocess.py:fuse_frames` | 다중 프레임 다수결이 체계적으로 정확한 통계적 이유 |
| `app/ocr/postprocess.py:_char_vote` | **어떤 프레임에도 없던 정답을 복원**하는 자리별 투표 |
| `app/ocr/postprocess.py:fuse_frames` 합의 보너스 | 독립 관측 수와 비율을 함께 보는 이유 — 결함 #16 |
| `tests/test_postprocess.py:test_no_reverse_correction_in_hangul_position` | 요구사항을 테스트로 고정하는 방법 |
| `tests/test_postprocess.py:test_fuse_char_vote_recovers_unseen_truth` | 융합 효과를 증명하는 테스트 설계 |

### 9장 — 영상 품질 평가 / 보정

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/pipeline/quality.py:laplacian_variance` | 2차 미분 분산이 선명도 지표인 원리, **왜 평균이 아닌 분산** |
| `app/pipeline/quality.py:saturation_ratio` | 평균 밝기가 놓치는 "반 흰 반 검정" 실패를 잡는 지표 |
| `app/pipeline/quality.py:assess` | 가중합 + **곱셈 벌점** 결합 설계 근거 |
| `app/pipeline/quality.py:_failure_reason` | 흰 번호판 오반려 수정 — 결함 #10 |
| `app/pipeline/quality.py:_blur_score` | 선형 대신 포화 곡선 `x/(x+k)` 을 쓰는 이유 |
| `app/pipeline/quality.py:DEFAULT_PLATE_PX_MIN` 주석 | 72px 하한의 **광학적 유도** (7자 x 9px + 여백) |
| `app/pipeline/rectify.py` 모듈 docstring | 기하→대비→샤프닝 **순서**가 중요한 이유 |
| `app/pipeline/rectify.py:estimate_skew_angle` | Hough 선분 각도의 **가중 중앙값** (이상치 강건성) |
| `app/pipeline/rectify.py:estimate_skew_angle` 경계 필터 | "경계에 닿음" vs "경계를 따라 달림" 구분 — 결함 #8 |
| `app/pipeline/rectify.py:_normalize_hough_output` | OpenCV 4.x/5.x API 차이 흡수 — 결함 #9 |
| `app/pipeline/rectify.py:order_quad` | 좌표 합/차로 4점을 정렬하는 기하학적 트릭 |
| `app/pipeline/rectify.py:perspective_correct` | 출력 크기를 **최댓값**으로 잡아 정보 손실을 막는 선택 |
| `app/pipeline/rectify.py:apply_clahe` | BGR 각 채널 대신 **LAB 의 L 채널**에만 적용하는 이유 |
| `app/pipeline/rectify.py:sharpen` | 언샤프 마스킹 원리 (`원본 + amount x 고주파`) |
| `app/pipeline/rectify.py:resize_for_ocr` | **오른쪽 패딩**이 CTC 에 무해한 이유 |

### 10장 — 파이프라인 통합 / 폴백 전략

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/pipeline/lpr_pipeline.py:LprPipeline.run` | 검출→보정→OCR→후처리 오케스트레이션 전체 흐름 |
| `app/pipeline/lpr_pipeline.py:_attempt_score` | **후보 경쟁** — 형식 유효성을 1순위로 두는 운영적 근거 |
| `app/pipeline/lpr_pipeline.py:MAX_OCR_CANDIDATES` 주석 | OCR 이 검출보다 싸기 때문에 가능한 전략 |
| `app/ocr/recognizer.py:create_recognizer` | **팩토리가 예외를 흡수**해야 "500 금지" 가 구조적으로 보장되는 이유 |
| `app/ocr/recognizer.py:PlateRecognizer.exposed_name` | `-fallback` 접미사 생성을 한 곳에 모으는 설계 |
| `app/ocr/paddle_adapter.py:get_paddle_ocr` | 버전별 생성자 인자 차이를 순차 시도로 흡수 |
| `app/ocr/paddle_adapter.py:_parse_paddle_output` | PaddleOCR 2.x/3.x 출력 형식 양쪽 파싱 |
| `app/ocr/paddle_adapter.py` 모듈 docstring | **선택 의존성 제외 결정의 3가지 근거** |
| `app/routers/lpr.py:normalize_detector` 외 | 잘못된 파라미터를 422 가 아니라 기본값으로 처리하는 이유 |
| `app/routers/lpr.py:exposed_model_names` | 조기 반환 경로의 이름 불일치 — 결함 #19 |
| `app/main.py:_contract_safe_response` | **경로별 계약 준수 실패 응답** — 500 을 복구 가능한 형태로 변환 |
| `app/main.py:validation_exception_handler` | 422 를 유지할 경로와 200 으로 바꿀 경로의 판단 기준 |
| `app/routers/health.py:_resolved_name` | 헬스체크가 모델을 로드하면 안 되는 이유 |

### 11장 — 예지보전: 특징 공학

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/pdm/features.py` 모듈 docstring | 현재값이 아니라 **추세와 변동성**이 고장을 예측하는 이유 |
| `app/pdm/features.py:build_feature_frame` | **전방 채움 vs 선형 보간** — 선형 보간이 가짜 추세를 만드는 문제 |
| `app/pdm/features.py:_forward_backward_fill` | `np.maximum.accumulate` 로 pandas 없이 ffill 구현 |
| `app/pdm/features.py:SENSOR_RANGES` 주석 | 각 센서 정상 범위의 **도메인 근거** (법정 기준 포함) |
| `app/pdm/features.py:moving_average` | 끝단을 버리지 않는 누적합 구현 — 최신 추세 보존 |
| `app/pdm/features.py:trend_slope` | "마지막-첫 값" 대신 회귀 기울기를 쓰는 노이즈 강건성 |
| `app/pdm/features.py:aggregate_features` | last/mean/std/slope 4종이 **정비사의 관찰과 일치**하는 이유 |
| `app/pdm/features.py:aggregate_features` 파생 블록 | **타이어 불균형** — 평균만 보면 펑크를 놓치는 사례 |
| `app/pdm/features.py:feature_vector` | 정렬 키 순서로 열 순서 불일치 사고를 구조적으로 막기 |
| `app/pdm/features.py:make_windows` | 데이터 부족 차량을 거부하지 않고 패딩하는 운영적 판단 |

### 12장 — 예지보전: 이상탐지

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/pdm/anomaly.py` 모듈 docstring | 고장 데이터 희소성 → **단일 클래스 학습**, 4방법 비교표 |
| `app/pdm/anomaly.py:statistical_anomaly` | 절대 임계 + z-score 를 **모두** 써야 하는 이유 |
| `app/pdm/anomaly.py:statistical_anomaly` 결합 블록 | z-score 는 올리기만 — **위험 고정 차량 역전** 수정 — 결함 #11 |
| `app/pdm/anomaly.py:RISK_THRESHOLDS` 주석 | 센서별 **방향성**(과압 vs 저압)이 심각도를 바꾸는 이유 |
| `app/pdm/anomaly.py:sklearn_anomaly` | IsolationForest `score_samples` 부호 반전 (**실무 빈출 버그**) |
| `app/pdm/anomaly.py:build_autoencoder` | 병목이 이상탐지에 쓰이는 원리, `latent_dim=8` 근거 |
| `app/pdm/anomaly.py:autoencoder_anomaly` | 체크포인트에 **정규화 통계 + 임계값**을 저장해야 하는 이유 |
| `app/pdm/anomaly.py:train_isolation_forest` | `contamination` 을 실제 고장률에 맞추는 근거 |
| `app/pdm/anomaly.py:train_one_class_svm` | OneClassSVM 의 스케일 민감성과 Pipeline 캡슐화 |
| `app/pdm/anomaly.py:_threshold_reason` | "이상입니다" 만으로는 조치 불가 — 설명 가능성의 운영 가치 |

### 13장 — 예지보전: 고장 분류와 RUL

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/pdm/failure.py` 모듈 docstring | 이상탐지 vs 고장분류의 역할 차이, 규칙 기반이 유효한 이유 |
| `app/pdm/failure.py:diagnose_engine` | 수준 + 추세 + 변동성 3신호 결합 (냉각 계통 고장 패턴) |
| `app/pdm/failure.py:diagnose_battery` | 전압과 SOH 를 **곱이 아니라 max** 로 결합하는 이유 |
| `app/pdm/failure.py:diagnose_tire` | 불균형 > 평균 — 펑크와 공기 부족의 구분 |
| `app/pdm/failure.py:diagnose_transmission` | **파생 특징의 가치** — rpm/speed 비율로 슬립 간접 탐지 |
| `app/pdm/failure.py:_score_from_risk` | 선형 대신 `(1-r)^0.75` 곡선으로 저위험 구간을 펼치는 이유 |
| `app/pdm/failure.py:_status_from_risk` | 4단계 경계값이 **운영 조치와 1:1 대응**하는 설계 |
| `app/pdm/failure.py:rule_based_failure` | 데이터 없음을 0 위험이 아니라 중립값으로 보고하는 이유 |
| `app/pdm/failure.py:sklearn_failure` | **모델 확률 + 규칙 사유 하이브리드** — 설명 가능성 유지 |
| `app/pdm/failure.py:sklearn_failure` classes 블록 | `model.classes_` 순서 불일치가 조용히 틀리는 문제 |
| `app/pdm/failure.py:train_failure_classifier` | `class_weight="balanced"` — 95% 정확도 함정 |
| `app/pdm/rul.py:linear_rul` | 선형 외삽의 **3가지 경계 처리**(초과/정체/개선) |
| `app/pdm/rul.py` 모듈 docstring | 소모성 부품에 선형이 잘 맞는 이유, NASA C-MAPSS 사례 |
| `app/pdm/rul.py:estimate_sample_interval_days` | 간격 추정에 **중앙값**을 쓰는 이유 (수집 중단 구간) |
| `app/pdm/rul.py:build_lstm_rul` | 마지막 은닉 상태만 쓰는 이유, 출력 ReLU 를 **안 쓰는** 이유 |
| `app/pdm/rul.py:lstm_rul` | `rul_scale` 을 체크포인트에 저장해야 하는 이유 |
| `app/pdm/rul.py:_linear_confidence` | 데이터 수 / 추세 명확성 / 완전성으로 신뢰도를 구성 |

### 14장 — 예지보전: 앙상블 서비스

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/pdm/service.py:compute_health_score` | 3신호 가중치 근거 + RUL 의 **로그 스케일** 변환 |
| `app/pdm/service.py:decide_risk_level` | **안전 상한(safety override)** — 가중 평균이 치명적 고장을 가리는 문제 |
| `app/pdm/service.py:build_recommendations` | 상태 + RUL 두 조건 — "아직 정상이지만 곧 아닌" 구간 — 결함 #12 |
| `app/pdm/service.py:build_recommendations` 상한 블록 | 상태 기반 기한 상한 — CRITICAL 을 730일 뒤로 미루는 모순 — 결함 #13 |
| `app/pdm/service.py:_predict_impl` 빈 데이터 블록 | 센서 고장 차량이 "가장 건강" 으로 집계되는 문제 — 결함 #14 |
| `app/pdm/service.py:PdmService._neutral_result` | 예측 실패 시 100점도 CRITICAL 도 아닌 **중립값**이 정직한 이유 |
| `app/pdm/service.py:MAINTENANCE_ACTIONS` | 추세 기반 거리 vs 제조사 권장 주기 중 **보수적인 쪽** 선택 |

### 15장 — 합성 데이터 생성

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/train/synth_plate.py` 모듈 docstring | 번호판 데이터가 희소한 이유, 합성 사전학습 + 미세조정 레시피 |
| `app/train/synth_plate.py:PLATE_STYLES` | 4종 규격(신형/사업용/구형/전기차)의 실제 치수 비례 |
| `app/train/synth_plate.py:_draw_tracked` | 자간 균등 분배로 번호판 서체를 재현 + 문자별 bbox 획득 |
| `app/train/synth_plate.py:augment_perspective` | 게이트 카메라 설치각을 호모그래피로 모사 |
| `app/train/synth_plate.py:augment_motion_blur` | **방향성 커널** vs 가우시안 — 주행 블러의 실제 형태 |
| `app/train/synth_plate.py:augment_lighting` | 2D 가우시안 밝기 맵이 전역 이진화를 무력화시키는 효과 |
| `app/train/synth_plate.py:augment_jpeg` | CCTV 경로의 블록 아티팩트를 학습에 포함해야 하는 이유 |
| `app/train/synth_plate.py:compose_scene` | **방해 에지(distractor)** 를 넣어야 검출기가 텍스처를 배우는 이유 |
| `app/train/synth_plate.py:generate_sample` | 실제 도로 분포를 반영한 스타일 가중치 |
| `app/train/synth_plate.py:random_plate_text` | 전역 `random` 대신 `Random(seed)` 로 재현성 확보 |

### 16장 — 학습 / 배포 최적화

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `app/train/train_detector.py:train` | 번호판 특화 증강 — **mosaic/fliplr 금지**의 명확한 근거 |
| `app/train/train_detector.py` 모듈 docstring | 단일 클래스에서 `box=8.0, cls=0.3` 가중치 조정 |
| `app/train/train_crnn.py` 모듈 docstring | **CTC 학습 4대 함정** (입력 형태/길이 조건/zero_infinity/blank 일치) |
| `app/train/train_crnn.py:train` | 기울기 클리핑, `ReduceLROnPlateau`, 비유한 손실 배치 건너뛰기 |
| `app/train/train_crnn.py:evaluate` | **완전 일치율을 EarlyStopping 기준**으로 쓰는 운영적 이유 |
| `app/train/train_crnn.py:load_image_gray` | 학습/추론 전처리 공유로 분포 불일치를 막기 |
| `app/train/export_onnx.py` 모듈 docstring | PyTorch vs ONNX 비교표, 내보내기 4대 필수 사항 |
| `app/train/export_onnx.py:_verify_onnx` | **"조용히 틀리는" ONNX 변환**을 수치 비교로 잡아내기 |
| `app/train/export_onnx.py:export_detector` | YOLO 는 ultralytics export 를 써야 하는 이유(후처리 결합) |
| `app/train/quantize.py` 모듈 docstring | **동적 vs 정적 양자화 비교표**, CRNN 에 동적이 적합한 이유 |
| `app/train/quantize.py:_compare_outputs` | **argmax 일치율**이 실질적 정확도 보존 지표인 이유 |
| `app/train/quantize.py:quantize_onnx` | ONNX 경로가 Conv 도 양자화해 YOLO 에 유효한 이유 |
| `app/train/benchmark.py` 모듈 docstring | **측정 방법론 4원칙** (워밍업/p95/지표분리/RSS 증분) |
| `app/train/benchmark.py:current_rss_mb` | psutil 없이 `/proc/self/statm` 으로 메모리 측정 |
| `app/train/benchmark.py:build_eval_set` | 합성 데이터 벤치마크의 **한계 명시** (상대 비교용) |

### 17장 — 테스트 전략

| 인용 코드 | 핵심 교육 포인트 |
|---|---|
| `tests/test_api.py` 상단 `importorskip` | 의존성 유무에 따라 테스트를 스킵해 CI 를 깨지지 않게 하기 |
| `tests/test_api.py:test_no_endpoint_returns_500` | **계약서 핵심 요구사항을 테스트로 고정** |
| `tests/test_api.py:assert_lpr_contract` | 계약 스키마 검증을 헬퍼로 추출해 중복 제거 |
| `tests/test_api.py:client` fixture | module 범위 fixture 로 앱 생성 비용 절감 |
| `tests/test_quality.py:DARK_TONE` 주석 | **테스트 픽스처가 현실을 반영해야 하는 이유** (0/255 포화 함정) |
| `tests/test_quality.py:test_clipping_penalty_lowers_score` | "선명하지만 포화된" 영상을 구분하는 테스트 설계 |
| `tests/test_ctc_decoder.py:make_probs` | 모델 없이 확률 행렬을 직접 구성해 디코더만 테스트 |
| `tests/test_ctc_decoder.py:test_collapse_order_matters` | 알고리즘 순서 의존성을 테스트로 고정 |
| `tests/test_ctc_decoder.py:test_blank_index_matches_charset` | 모듈 간 불변식(blank 인덱스 일치)을 테스트로 보호 |
| `tests/test_postprocess.py:test_use_chars_are_all_complete_hangul` | 데이터 불변식 테스트 |
| `tests/test_postprocess.py:test_private_use_chars_have_no_jongseong` | 과도하게 엄격한 단정의 수정 사례(특수 번호판 예외) |

---

## 4. 실행 순서 요약 (교재 실습용)

```bash
# 0) 최소 설치로 서비스 기동 — 가중치 없이 전체 동작 확인
uvicorn app.main:app --port 8000
curl -s localhost:8000/health | python3 -m json.tool      # detector: yolo-fallback

# 1) 테스트 통과 확인 (네트워크 불필요)
pytest -q                                                  # 132 passed

# 2) 폴백 성능 측정
python -m app.train.benchmark --samples 15                 # recall 0.933 / exact 0.600

# 3) 합성 데이터 생성 → 학습 → 내보내기 → 양자화 → 재측정
python -m app.train.synth_plate --out ./data/synth --count 2000
python -m app.train.synth_plate --out ./data/ocr --count 20000 --no-scene
python -m app.train.train_detector --data ./data/synth/data.yaml --epochs 50
python -m app.train.train_crnn --data ./data/ocr --epochs 30
python -m app.train.export_onnx --crnn ./models/crnn_ctc_kr.pt
python -m app.train.quantize --crnn ./models/crnn_ctc_kr.pt
python -m app.train.benchmark --samples 50                 # 폴백 대비 개선 확인

# 4) 가중치가 생긴 뒤 헬스체크 — -fallback 접미사가 사라진다
curl -s localhost:8000/health | python3 -m json.tool
```
