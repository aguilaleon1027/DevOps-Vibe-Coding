<<<<<<< HEAD
# DevOps-Vibe-Coding
=======
# Workout Recommendation MLOps Pipeline

PyTorch 및 FastAPI를 활용한 딥러닝 기반 맞춤형 운동 프로그램 추천 API 서버 파이프라인.

이 저장소는 사용자의 개인적 특성(나이, 성별, 목표 체중/체지방량/근육량)을 바탕으로 머신러닝 모델이 직접 학습하고, 이 정보를 기반으로 FastAPI를 통해 최적의 운동 프로그램을 추천하도록 설계되었습니다.

## 🚀 시작하기 (Getting Started)

### 1. 가상환경 및 패키지 설치
Python 3.10+ 환경을 권장합니다.

```powerShell
# 가상환경 생성 및 활성화
python -m venv venv
.\venv\Scripts\activate

# 의존성 패키지 설치
pip install -r requirements.txt
```

### 2. 가상 데이터 생성 및 ML 모델 학습 (MLOps)
다음 스크립트를 실행하면 가상의 사용자 데이터를 2000개 생성하고 PyTorch 모델 학습을 시작합니다.
또한 MLflow를 통해 학습 내역이 로컬 SQLite에 추적 및 로깅됩니다.

```powerShell
# 프로젝트 최상단 디렉토리에서 실행
python ml_pipeline/train.py
```

학습이 정상적으로 완료되면 다음 파일들이 생성됩니다:
- `data/workout_data.csv` (학습에 사용된 가상 데이터)
- `weights/best_model.pth` (PyTorch 모델 가중치)
- `models/scaler.joblib` (데이터 스케일러)
- `mlruns/` (MLflow 로깅 정보)
- `mlflow.db` (MLflow 백엔드 스토어)

**MLflow UI로 파라미터 확인하기:**
```powerShell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
브라우저에서 `http://localhost:5000` 로 접속하시면 Loss 추이, Accuracy 및 학습 파라미터를 시각적으로 확인할 수 있습니다.

### 3. API 서버 실행
학습이 완료된 후 FastAPI 앱을 구동합니다.
```powerShell
uvicorn app.main:app --reload
```

- **API Documentation**: 브라우저에서 `http://localhost:8000/docs` 에 접속하시면 Swagger UI를 통해 곧바로 API를 테스트 해볼 수 있습니다!

## 📌 기능 명세서
- **POST** `/api/v1/recommend`: 사용자의 목표를 반영한 운동 프로그램 카테고리(Cardio, Hypertrophy 등)와 설명을 JSON 객체로 반환.
- **GET** `/api/v1/health`: API 서버의 헬스체크.

---
**Note:** 이 프로젝트는 기본 뼈대 역할의 템플릿 코드입니다. 향후 실제 데이터 연결, 모델 튜닝(Hyperparameter 튜닝), Dockerized 배포(CI/CD) 구성을 추가하여 고도화할 수 있습니다.
>>>>>>> 7eca12e (first commit)
