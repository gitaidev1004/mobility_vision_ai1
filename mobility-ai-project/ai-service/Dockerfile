# AI 추론 서비스 (FastAPI + PyTorch CPU)
# 빌드:  docker build -t mobility-ai-service ./ai-service
# 실행:  docker run -p 8000:8000 mobility-ai-service
FROM python:3.12-slim

# OpenCV 가 필요로 하는 최소 런타임 라이브러리만 설치한다.
# opencv-python-headless 를 쓰므로 GUI 관련 패키지는 넣지 않는다.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 레이어를 먼저 만들어 소스 변경 시 재설치를 피한다.
COPY requirements.txt .
# PyTorch 는 CPU 전용 휠을 별도 인덱스에서 받는다(이미지 용량을 크게 줄인다).
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app

# 가중치는 볼륨으로 주입한다. 없으면 폴백 추론으로 동작한다.
RUN mkdir -p /app/models

ENV AI_HOST=0.0.0.0 \
    AI_PORT=8000 \
    AI_TORCH_THREADS=4 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
