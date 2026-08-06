from fastapi import FastAPI, UploadFile, File, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api.routes import predict, chat, search
from api.services.metrics import get_metrics

app = FastAPI(title="Car Sales Intelligence API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(chat.router, prefix="/chat", tags=["Chatbot"])
app.include_router(search.router, prefix="/search", tags=["Search"])

@app.get("/")
def root():
    return {"status": "Ok", "services": "Car Sales Intelligence API"}

@app.get("/metrics")
async def metrics():
    return Response(content=get_metrics(), media_type="text/plain")

@app.get("/business-metrics")
def business_metrics()

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False)