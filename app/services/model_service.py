import torch
import joblib
import os
import numpy as np
import sys

# Dynamically add ml_pipeline to python path so we can import WorkoutRecommender
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ml_pipeline"))
from model import WorkoutRecommender
from app.core.config import settings

class InferenceService:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = None
        self.scaler = None
        
        # Output label mapping based on our dummy data logic
        self.program_map = {
            0: ("Cardio Focus", "체중 감량을 위한 유산소 위주의 프로그램"),
            1: ("Hypertrophy Focus", "근육량 증가를 위한 웨이트 트레이닝 프로그램"),
            2: ("Endurance Focus", "심폐지구력 및 체력 증진을 위한 서킷 트레이닝"),
            3: ("Mobility & Rehab Focus", "유연성 증가 및 관절 가동성 등 교정을 위한 프로그램"),
            4: ("Maintenance Focus", "현재 체성분 유지를 위한 전신 밸런스 운동 프로그램")
        }
        self._load_model()
        
    def _load_model(self):
        try:
            self.model = WorkoutRecommender()
            if os.path.exists(settings.MODEL_WEIGHTS_PATH):
                self.model.load_state_dict(torch.load(settings.MODEL_WEIGHTS_PATH, map_location=self.device))
                self.model.eval()
            else:
                print(f"Warning: Model weights not found at {settings.MODEL_WEIGHTS_PATH}.")
                
            if os.path.exists(settings.SCALER_PATH):
                self.scaler = joblib.load(settings.SCALER_PATH)
            else:
                print(f"Warning: Scaler not found at {settings.SCALER_PATH}.")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")

    def predict(self, input_features: list) -> dict:
        features_np = np.array(input_features).reshape(1, -1)
        if self.scaler:
            features_np = self.scaler.transform(features_np)
            
        inputs_tensor = torch.tensor(features_np, dtype=torch.float32)
        
        with torch.no_grad():
            outputs = self.model(inputs_tensor)
            _, predicted_class = torch.max(outputs, 1)
            
        pred_idx = predicted_class.item()
        name, desc = self.program_map.get(pred_idx, ("Unknown", "No description available"))
        
        return {
            "recommended_program": pred_idx,
            "program_name": name,
            "description": desc
        }

inference_service = InferenceService()
