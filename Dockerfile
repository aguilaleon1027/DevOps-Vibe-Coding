# 사용할 베이스 이미지 (가벼운 slim 환경 사용)
FROM python:3.10-slim AS builder

# 작업 디렉토리 설정
WORKDIR /code

# 모델 가중치를 찾을 수 있도록 환경 변수 설정
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_WEIGHTS_PATH="weights/best_model.pth" \
    SCALER_PATH="models/scaler.joblib"

# 시스템 패키지 업데이트 및 C 컴파일러 설치 (머신러닝 라이브러리 설치 의존성)
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# 요구사항 파일 복사
COPY requirements.txt .

# 패키지 설치 최적화
# (PyTorch의 경우 기본 이미지 용량이 5GB를 넘어가므로, CPU 버전의 가벼운 PyTorch로 명시적 설치합니다.)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 애플리케이션 코드 및 학습된 가중치 파일 복사
# (.dockerignore 규칙에 의해 불필요한 venv, __pycache__, 데이터 등은 무시됩니다)
COPY ./app /code/app
COPY ./weights /code/weights
COPY ./models /code/models

# 보안을 위해 컨테이너 구동 사용자 변경 (root 권한 탈취 방지)
RUN useradd -m appuser && chown -R appuser:appuser /code
USER appuser

# FastAPI 포트 노출
EXPOSE 8000

# API 서버를 운영 모드로 실행
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
