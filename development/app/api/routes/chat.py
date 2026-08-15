from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
from api.services.chatbot import CarChatbot

router = APIRouter()

# Define request and response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    intent: str
    entities: Dict[str, Any]
    session_id: str
    timestamp: str

# Initialize the chatbot service
chatbot = CarChatbot()

@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the car sales assistant"""

    try:
        # Generate Session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Get response from the chatbot
        result = chatbot.chat(request.message, session_id)

        return ChatResponse(
            response=result['response'],
            intent=result['intent'],
            entities=result['entities'],
            session_id=session_id,
            timestamp=result['timestamp']
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset")
async def reset_session(session_id: str):
    """Reset chat session"""
    try:
        chatbot.reset_context(session_id)
        return {"status": "success", "message": f"Session {session_id} reset"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/intents")
async def list_intents():
    """List available intents"""
    return {
        "intents": list(chatbot.intent_patterns.keys()),
        "description": "Supported intents for car chat assistant"
    }