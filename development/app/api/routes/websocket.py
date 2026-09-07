from fastapi import WebSocket, WebSocketDisconnect, Depends, APIRouter, Query
import json
import jwt
import logging
from typing import Optional
from pathlib import Path
import os
from dotenv import load_dotenv
from api.routes.chat import get_chatbot

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.user_connections: dict[str, str] = {}  # Maps user_id to connection_id
        self.user_sessions: dict[str, dict] = {}  # Maps user_id to session data

    async def connect(self, websocket: WebSocket, user_id: str):
        connection_id = str(id(websocket))
        self.active_connections[connection_id] = websocket
        self.user_connections[user_id] = connection_id
        self.user_sessions[user_id] = f"ws_session_{user_id}_{id(websocket)}"
        return connection_id

    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        # Remove user_id mapping
        for user_id, conn_id in list(self.user_connections.items()):
            if conn_id == connection_id:
                del self.user_connections[user_id]
                if user_id in self.user_sessions:
                    del self.user_sessions[user_id]
                break

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.user_connections:
            connection_id = self.user_connections[user_id]

            if connection_id in self.active_connections:
                await self.active_connections[connection_id].send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections.values():
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, 
    token: Optional[str] = Query(None)):

    # Accept the connection first
    await websocket.accept()
    logger.info("🔗 WebSocket connection accepted")

    # Initialize variables
    user_id = None
    connection_id = None

    try:
        if not token:
            logger.warning("❌ No token provided in WebSocket connection")
            await websocket.send_text(json.dumps({"error": "Token is required"}))
            await websocket.close(code=1008)
            return

        # Decode and validate the JWT token
        try:
            payload = jwt.decode(
                token,
                os.getenv("AUTH_JWT_SECRET_KEY"),
                algorithms=[os.getenv("AUTH_JWT_ALGORITHM")])
            user_id = payload.get("sub")
            user_role = payload.get("role")

            if not user_id:
                logger.warning("❌ Invalid token provided")
                await websocket.send_text(json.dumps({"error": "Invalid token"}))
                await websocket.close(code=1008)
                return

            logger.info(f"✅ WebSocket connection established for user_id: {user_id}, role: {user_role}")

        except jwt.ExpiredSignatureError:
            logger.warning("❌ Token expired for WebSocket connection")
            await websocket.send_text(json.dumps({"error": "Token expired"}))
            await websocket.close(code=1008)
            return

        except jwt.InvalidTokenError:
            logger.warning(f"❌ Invalid token: {e}")
            await websocket.send_text(json.dumps({"error": "Invalid token"}))
            await websocket.close(code=1008)
            return

        # Register the connection
        connection_id = await manager.connect(websocket, user_id)
        session_id = manager.user_sessions.get(user_id, user_id)

        # Get chatbot instance
        try:
            chatbot = get_chatbot()
            logger.info("✅ Chatbot instance retrieved successfully")

        except Exception as e:
            logger.error(f"❌ Failed to retrieve chatbot instance: {e}")
            chatbot = None

        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "connection",
            "status": "connected",
            "user_id": user_id,
            "role": user_role,
            "session_id": session_id,
            "message": "Welcome to Car Sales Intelligence! How can I help you today?"
        }))

        logger.info(f"📨 Welcome message sent to {user_id}")

        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                logger.info(f"📥 Received message from {user_id}: {data}")

                try:
                    message = json.loads(data)
                except json.JSONDecodeError:
                    logger.warning(f"❌ Invalid JSON format from {user_id}: {data}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "Invalid JSON format"
                    }))
                    continue

                # Process message based on type
                message_type = message.get("type", "message")

                if message_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                    logger.debug(f"🏓 Pong sent to {user_id}")

                elif message_type == "chat":
                    content = message.get("content", "")
                    logger.info(f"💬 Chat message from {user_id}: {content}")

                    if not content:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "error": "Empty message content"
                        }))
                        continue

                    # Process chat message using chatbot
                    try:
                        if chatbot:
                            result = chatbot.chat(content, session_id)
                            response = result.get('response', "I'm not sure how to respond to that.")
                            intent = result.get('intent', 'general')
                            entities = result.get('entities', {})
                            source = result.get('source', 'unknown')
                            timestamp = result.get('timestamp', '')

                            logger.info(f"🤖 Chatbot response for {user_id}: {response} (Intent: {intent}, Source: {source})")

                            # Send response back to client
                            await websocket.send_text(json.dumps({
                                "type": "chat_response",
                                "content": response,
                                "intent": intent,
                                "entities": entities,
                                "source": source,
                                "timestamp": timestamp
                            }))
                        else:
                            # Fallback if chatbot is not available
                            await websocket.send_text(json.dumps({
                                "type": "chat_response",
                                "content": "Chat service is currently unavailable. Please try again later.",
                                "intent": "error",
                                "entities": {},
                                "source": "error"
                            }))

                    except Exception as e:
                        logger.error(f"❌ Chatbot error for {user_id}: {e}")
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "error": "Failed to process chat message"
                        }))

                elif message_type == "reset":
                    # Reset conversation context
                    if chatbot and hasattr(chatbot, 'reset_context'):
                        chatbot.reset_context(session_id)
                    await websocket.send_text(json.dumps({
                        "type": "reset_ack",
                        "message": "Conversation context reset"
                    }))

                else:
                    await websocket.send_text(json.dumps({
                        "type": "echo",
                        "data": message
                    }))

            except WebSocketDisconnect:
                logger.info(f"🔌 WebSocket disconnected for user: {user_id}")
                manager.disconnect(connection_id)
                break

            except Exception as e:
                logger.error(f"❌ Error processing message from {user_id}: {e}")

                try:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "error": "Internal server error"
                    }))
                except:
                    break

    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected for user: {user_id}")
    except Exception as e:
        logger.error(f"❌ Websocket error: {str(e)}")
        try:
            await websocket.close(code=1011)  # Internal Error
        except:
            pass

# Function to get chatbot status
@router.get("/ws/status")
async def get_websocket_status():
    """Get Websocket connection status"""
    try:
        chatbot = get_chatbot()
        return {
            "status": "healthy",
            "active_connections": len(manager.active_connections),
            "active_users": len(manager.user_connections),
            "chatbot_available": chatbot is not None,
            "chatbot_type": "production" if chatbot and hasattr(chatbot, 'use_llm') and chatbot.use_llm else "local"
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }