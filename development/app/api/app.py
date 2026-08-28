from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from api.routes import predict, chat, search, auth, websocket
from api.services.metrics import get_metrics, get_business_metrics
import logging

# Logger configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Environment Configuration
# ============================================
USE_PRODUCTION = os.getenv("USE_PRODUCTION_SERVICES", "false").lower() == "true"

app = FastAPI(
    title="Car Sales Intelligence API",
    version="1.0.0",
    description="Car Sales API with Local & Production support"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Include Routers
# ============================================
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(chat.router, prefix="/chat", tags=["Chatbot"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(websocket.router, prefix="", tags=["WebSocket"])

# ============================================
# Health & Metrics Endpoints
# ============================================
@app.get("/health")
def health_check():
    return {"status": "✅ healthy", "message": "🏎️ Car Sales Intelligence API is running."}

@app.get("/metrics")
async def metrics():
    return Response(content=get_metrics(), media_type="text/plain")

@app.get("/business-metrics")
def business_metrics():
    """JSON business KPIs for the frontend dashboard."""
    return get_business_metrics(format_type="json")

@app.get("/env")
async def get_environment():
    """Get current environment configuration"""
    return {
        "environment": "production" if USE_PRODUCTION else "development",
        "use_production": USE_PRODUCTION,
        "services": {
            "predictor": "Production (MinIO)" if USE_PRODUCTION else "Local",
            "chatbot": "Production (LLM/MinIO)" if USE_PRODUCTION else "Local",
            "cv_search": "Production (MinIO)" if USE_PRODUCTION else "Local"
        }
    }

# ============================================
# Startup Event
# ============================================
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup based on environment"""
    
    if USE_PRODUCTION:
        print("🚀 Initializing Production Services...")
        
        # Import production services
        try:
            from api.services.predictor_production import get_predictor as get_prod_predictor
            from api.services.cv_search_production import get_cv_search as get_prod_cv_search
            from api.services.chatbot_production import get_chatbot as get_prod_chatbot
            
            # Pre-load services to avoid cold start
            get_prod_predictor()
            print("✅ Production Predictor initialized")
            
            get_prod_cv_search()
            print("✅ Production CV Search initialized")
            
            get_prod_chatbot()
            print("✅ Production Chatbot initialized")
            
            print("✅ All production services initialized successfully.")
            
        except ImportError as e:
            print(f"⚠️ Production service import failed: {e}")
            print("💻 Falling back to development mode...")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize production services: {e}")
            print("💻 Falling back to development mode...")
    
    else:
        print("💻 Running in Development Mode (local services)")

# ============================================
# Main Entry Point
# ============================================
if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload for production
    )