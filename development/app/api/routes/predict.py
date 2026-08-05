from fastapi import APIRouter, HTTPException
from api.models import PredictRequest, PredictResponse, AvailableModelsResponse
from api.services.predictor import ModelPredictor

router = APIRouter()
predictor = ModelPredictor()

@router.get("/models", response_model=AvailableModelsResponse)
async def list_available_models():
    try:
        return predictor.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving available models: {str(e)}")

@router.post("/", response_model=PredictResponse)
async def predict(request: PredictRequest):
    try:
        result_dict = predictor.predict(request)
        return PredictResponse(**result_dict)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")