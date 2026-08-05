from pydantic import BaseModel
from typing import Optional, List

class PredictRequest(BaseModel):
    model_type: str # "Sales" or "Quantity"
    model_name: str
    features: dict

class PredictResponse(BaseModel):
    prediction: float
    model_name: str
    confidence: Optional[float] = None

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