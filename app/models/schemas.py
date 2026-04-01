from pydantic import BaseModel, Field

class WorkoutRecommendationRequest(BaseModel):
    age: int = Field(..., ge=15, le=100, description="Age of the user (15-100)")
    gender: int = Field(..., ge=0, le=1, description="Gender (0: Male, 1: Female)")
    target_weight: float = Field(..., gt=30, le=200, description="Target weight in kg")
    target_fat: float = Field(..., gt=0, le=50, description="Target body fat percentage")
    target_muscle: float = Field(..., gt=10, le=80, description="Target muscle mass in kg")

class WorkoutRecommendationResponse(BaseModel):
    recommended_program: int
    program_name: str
    description: str
