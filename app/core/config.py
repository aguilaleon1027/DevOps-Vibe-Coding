import os

class Settings:
    PROJECT_NAME = "Workout Recommendation API"
    VERSION = "1.0.0"
    API_V1_STR = "/api/v1"
    
    # Assuming the app is run from the root of the project:
    MODEL_WEIGHTS_PATH = os.getenv("MODEL_WEIGHTS_PATH", "weights/best_model.pth")
    SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")

settings = Settings()
