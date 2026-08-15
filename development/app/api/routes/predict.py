from fastapi import APIRouter, HTTPException
from api.models import PredictRequest, PredictResponse, AvailableModelsResponse
from api.services.predictor import ModelPredictor
from api.services.metrics import track_prediction

router = APIRouter()
predictor = ModelPredictor()
confidence_score = predictor._calculate_confidence_score

# Define available models to be used in the API
available_models = predictor.get_available_models()
default_sales_model = (
    available_models.get("sales_models", ["Xgboost"])[0]
    if available_models.get("sales_models")
    else "Xgboost"
)

@router.get("/models", response_model=AvailableModelsResponse)
async def list_available_models():
    try:
        return predictor.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving available models: {str(e)}")

@router.post("/", response_model=PredictResponse)
@track_prediction(model_type="sales", model_name= default_sales_model)
async def predict(request: PredictRequest):
    try:
        result = predictor.predict(request)

        # Standardize result extraction (dict or object)
        res_dict = result if isinstance(result, dict) else getattr(result, "__dict__", {})

        # Extract predicted values (supports single or 'both' predictions)
        pred_sales = res_dict.get("predicted_sales") or res_dict.get("sales") or res_dict.get("prediction")
        pred_qty = res_dict.get("predicted_quantity") or res_dict.get("quantity")

        # Calculate or retrieve confidence scores
        sales_conf = res_dict.get("sales_confidence") or confidence_score(pred_sales) if pred_sales is not None else None
        qty_conf = res_dict.get("quantity_confidence") or confidence_score(pred_qty) if pred_qty is not None else None

        model_type_val = res_dict.get(
            "model_type",
            getattr(request, "model_type", "sales")
        )

        return PredictResponse(
            model_type=model_type_val,
            predicted_sales=pred_sales,
            sales_confidence=sales_conf,
            predicted_quantity=pred_qty,
            quantity_confidence=qty_conf,
        )
        
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")