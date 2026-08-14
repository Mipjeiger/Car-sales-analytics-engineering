from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class PredictRequest(BaseModel):
    model_type: str = "both"  # Options: "sales", "quantity", "both"
    sales_model_name: Optional[str] = None
    quantity_model_name: Optional[str] = None
    # Fallback single model_name parameter for backward compatibility
    model_name: Optional[str] = None
    features: Dict[str, Any]


class AvailableModelsResponse(BaseModel):
    sales_models: List[str]
    quantity_models: List[str]


class PredictResponse(BaseModel):
    model_type: str
    predicted_sales: Optional[float] = None
    sales_confidence: Optional[float] = None
    predicted_quantity: Optional[float] = None
    quantity_confidence: Optional[float] = None


class ChatRequest(BaseModel):
    message: str
    context: Optional[List[str]] = None


class ChatResponse(BaseModel):
    response: str
    intent: str
    data: Optional[dict] = None


class SearchRequest(BaseModel):
    image_path: str
    k: int = 5


class SearchResponse(BaseModel):
    results: List[dict]
    query_brand: Optional[str] = None
    total_results: int
