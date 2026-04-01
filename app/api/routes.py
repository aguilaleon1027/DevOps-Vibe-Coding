from fastapi import APIRouter
from app.models.schemas import WorkoutRecommendationRequest, WorkoutRecommendationResponse
from app.services.model_service import inference_service

router = APIRouter()

@router.post("/recommend", response_model=WorkoutRecommendationResponse, summary="Get Workout Recommendation")
async def get_workout_recommendation(request_data: WorkoutRecommendationRequest):
    # Map input according to the training data structure
    features = [
        float(request_data.age),
        float(request_data.gender),
        request_data.target_weight,
        request_data.target_fat,
        request_data.target_muscle
    ]
    
    result = inference_service.predict(features)
    
    return WorkoutRecommendationResponse(**result)

@router.get("/health", summary="Health Check")
def health_check():
    return {"status": "healthy"}
