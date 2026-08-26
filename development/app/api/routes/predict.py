from fastapi import APIRouter, HTTPException, Depends
from api.models import PredictRequest, PredictResponse, AvailableModelsResponse
from api.services.predictor import ModelPredictor
from api.services.metrics import track_prediction
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

router = APIRouter()

# Determine which predictor to use
USE_PRODUCTION = os.getenv("USE_PRODUCTION_SERVICES", "false").lower() == "true"

# ==========================
# Predictor Factory
# ==========================
def get_predictor():
    """Get the appropriate predictor based on environment settings."""
    if USE_PRODUCTION:
        try:
            from api.services.predictor_production import get_predictor as get_prod_predictor
            predictor = get_prod_predictor()
            logger.info("✅ Using Production Predictor Service")

            return predictor

        except Exception as e:
            logger.warning(f"⚠️ Failed to load Production Predictor Service: {e}. Falling back to Development Predictor.")

    logger.info("💻 Using Local Predictor")
    return ModelPredictor()

# Intiailize predictor
predictor = get_predictor()

def get_confidence_score(value):
    """Calculate confidence score"""
    if hasattr(predictor, "_calculate_confidence_score"):
        return predictor._calculate_confidence_score(value)

# ==========================
# Get available models
# ==========================
def get_available_models():
    """Get available models from predictor"""
    if hasattr(predictor, 'get_available_models'):
        return predictor.get_available_models()

    # Fallback for local predictor
    return {
        "sales_models": ["XGBoost", "Random Forest", "Decision Tree", "CatBoost"],
        "quantity_models": ["XGBoost", "Random Forest", "Decision Tree", "CatBoost"]
    }

available_models = get_available_models()
default_sales_model = available_models.get("sales_models", ["XGBoost"])[0] if available_models.get("sales_models") else "XGBoost"

# =========================
# API Endpoints
# =========================

@router.get("/models", response_model=AvailableModelsResponse)
async def list_available_models():
    try:
        return predictor.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving available models: {str(e)}")

@router.get("/status")
async def get_predictor_status():
    """Get predictor status and source"""
    source = "Production (MinIO)" if USE_PRODUCTION else "Local"
    if USE_PRODUCTION:
        try:
            info = predictor.list_models() if hasattr(predictor, 'list_models') else {}
            return {
                "status": "✅ healthy",
                "source": source,
                "available models": info,
                "environment": "production" if USE_PRODUCTION else "development"
            }
        except:
            pass

    return {
        "status": "✅ healthy",
        "source": source,
        "available models": available_models,
        "environment": "production" if USE_PRODUCTION else "development"
    }

@router.post("/", response_model=PredictResponse)
@track_prediction(model_type="sales", model_name= default_sales_model)
async def predict(request: PredictRequest):
    try:
        # Use production or local predictor based on environment
        if USE_PRODUCTION and hasattr(predictor, 'predict_sales'):
            # Production predictor
            if request.model_type == "sales":
                result = predictor.predict_sales(request.features, model_name=request.model_name)
                pred_sales = result
                pred_qty = None
                sales_conf = get_confidence_score(pred_sales)

            elif request.model_type == "quantity":
                result = predictor.predict_quantity(request.features, model_name=request.model_name)
                pred_sales = None
                sales_conf = None
                pred_qty = result
                qty_conf = get_confidence_score(pred_qty)

            else:
                # Both sales and quantity predictions
                result = predictor.predict_sales(request.features, model_name=request.model_name)
                pred_sales = result
                pred_qty = predictor.predict_quantity(request.features, model_name=request.model_name)
                sales_conf = get_confidence_score(pred_sales)
                qty_conf = get_confidence_score(pred_qty)

        else:
            # Local predictor & Standardize result extraction
            result = predictor.predict(request)
            result_dict = result if isinstance(result, dict) else getattr(result, '__dict__', {})
            pred_sales = result_dict.get('predicted_sales') or result_dict.get('sales') or result_dict.get('prediction')
            pred_qty = result_dict.get('predicted_quantity') or result_dict.get('quantity')
            sales_conf = result_dict.get('sales_confidence') or get_confidence_score(pred_sales) if pred_sales is not None else None
            qty_conf = result_dict.get('quantity_confidence') or get_confidence_score(pred_qty) if pred_qty is not None else None

        return PredictResponse(
            model_type=request.model_type,
            predicted_sales=pred_sales,
            sales_confidence=sales_conf,
            predicted_quantity=pred_qty,
            quantity_confidence=qty_conf,
        )

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"❌ Invalid input: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Prediction failed: {str(e)}")