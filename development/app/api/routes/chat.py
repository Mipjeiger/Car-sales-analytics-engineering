from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uuid
import logging
import os

# ==========================
# Environment Configuration
# ==========================
USE_PRODUCTION = os.getenv("USE_PRODUCTION_SERVICES", "false").lower() == "true"

logger = logging.getLogger(__name__)
router = APIRouter()

# =========================
# Schemas
# =========================
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

# Chatbot instance - cached globally
_chatbot = None

# ========================
# Chatbot Initialization
# ========================
def get_chatbot():
    """Get the appropriate chatbot based on environment settings."""
    global _chatbot

    if _chatbot is not None:
        return _chatbot

    if USE_PRODUCTION:
        try:
            from api.services.chatbot_production import get_chatbot as get_prod_chatbot
            _chatbot = get_prod_chatbot()
            logger.info("✅ Using Production Chatbot Service")
            return _chatbot
        
        except Exception as e:
            logger.warning(f"⚠️ Failed to load Production Chatbot Service: {e}. Falling back to Development Chatbot.")

    # Fallback to local chatbot
    try:
        from api.services.chatbot import CarChatbot
        _chatbot = CarChatbot()
        logger.info("💻 Using Local Chatbot Service")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Local Chatbot Service: {e}")
        _chatbot = None

    return _chatbot

# ========================
# API Endpoints
# ========================
@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat with the car sales assistant."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        chatbot = get_chatbot()

        # Get response from the chatbot
        result = chatbot.chat(request.message, session_id)

        return ChatResponse(
            response=result.get('response', ''),
            intent=result.get('intent', 'general'),
            entities=result.get('entities', {}),
            session_id=session_id,
            timestamp=result.get('timestamp', '')
        )

    except Exception as e:
        logger.error(f"⚠️ Chatbot error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/session_chat')
async def get_session_chat():
    """Get chat history by session ID list if exists"""
    try:
        chatbot = get_chatbot()
        if hasattr(chatbot, 'get_session_history'):
            sessions = chatbot.get_session_history()
            return {
                'status': 'success',
                'session_count': len(sessions),
                'sessions': sessions
            }

        return {
            'status': 'success',
            'session_count': 0,
            'sessions': []
        }

    except Exception as e:
        logger.error(f"⚠️ Failed to retrieve session chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/intents')
async def list_intents():
    """List available intents"""
    try:
        chatbot = get_chatbot()
        if hasattr(chatbot, 'get_intents'):
            return chatbot.get_intents()
        if hasattr(chatbot, 'intent_patterns'):
            return {
                'intents': list(chatbot.intent_patterns.keys()),
                'description': 'Supported intents for car chat assistant'
            }
    except:
        pass
    
    return {
        'intents': ["price", "recommend", "sales", "specs", "compare", "greeting", "help"],
        "description": "Supported intents for car chat assistant"
    }

@router.get("/status")
async def get_chatbot_status():
    """Get chatbot status"""
    try:
        chatbot = get_chatbot()
        source = "Production (LLM/MinIO)" if USE_PRODUCTION else "Local"
        return {
            "status": "healthy",
            "source": source,
            "use_llm": getattr(chatbot, 'use_llm', False),
            "environment": "production" if USE_PRODUCTION else "development",
            "chatbot_loaded": chatbot is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }