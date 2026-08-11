from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from api.routes import predict, chat, search
from api.services.metrics import (
    get_metrics,
    get_business_metrics,
    PrometheusMiddleware,
)

app = FastAPI(title="Car Sales Intelligence API", version="1.0.0")

# 1. Register Prometheus Middleware (tracks all API traffic)
app.add_middleware(PrometheusMiddleware)

# 2. Register CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Include API Routers
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(chat.router, prefix="/chat", tags=["Chatbot"])
app.include_router(search.router, prefix="/search", tags=["Search"])

@app.get("/health")
def health_check():
    return {
        "status": "Ok",
        "services": "Car Sales Intelligence API",
        "health": "✅",
        "version": "1.0.0",
        "message": "Welcome to the Car Sales Intelligence API!",
    }

@app.get("/metrics")
async def metrics():
    """Prometheus scraping endpoint"""
    return Response(content=get_metrics(), media_type="text/plain")

@app.get("/business-metrics")
async def business_metrics():
    """Business metrics JSON endpoint"""
    return get_business_metrics(format_type="json")

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)