# --- Stage 1: Trainer (모델 학습 환경구축 및 훈련) ---
FROM python:3.10-slim AS trainer

WORKDIR /train_env

# 의존성 설치 환경 구성
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 머신러닝 파이프라인 코드 전체 복사 (app 코드는 제외 가능)
COPY ml_pipeline/ ./ml_pipeline/

# 💡 Docker Image Build 시점에 훈련 스크립트를 직접 실행하여 가중치 파일을 갓 구워냅니다!!
RUN python ml_pipeline/train.py


# --- Stage 2: Runner (가벼운 API 프로덕션 배포 파이프라인) ---
FROM python:3.10-slim AS runner

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_WEIGHTS_PATH="weights/best_model.pth" \
    SCALER_PATH="models/scaler.joblib"

# 파이썬 앱 실행에 필요한 라이브러리만 설치 (학습에만 쓰인 무거운 라이브러리는 제외해도 되나 여기선 단순화)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 1번 Stage (Trainer) 에서 새로 학습된 가중치와 스케일러만 '쏙 빼서' 현재 Stage 로 복사! 
COPY --from=trainer /train_env/weights/best_model.pth ./weights/best_model.pth
COPY --from=trainer /train_env/models/scaler.joblib ./models/scaler.joblib

# 앱 배포용 서비스 코드만 복사
COPY app/ ./app/

# 권한 및 포트 설정
RUN useradd -m appuser && chown -R appuser:appuser /code
USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
