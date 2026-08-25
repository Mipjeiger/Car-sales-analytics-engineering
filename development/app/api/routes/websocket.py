from fastapi import WebSocket, WebSocketDisconnect, Depends, APIRouter, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
import jwt
from typing import Optional
from pathlib import Path
import os
from dotenv import load_dotenv

# Base directory setup
BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

router = APIRouter()
security = HTTPBearer()

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.user_connections: dict[str, str] = {}  # Maps user_id to connection_id

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        connection_id = str(id(websocket))
        self.active_connections[connection_id] = websocket
        self.user_connections[user_id] = connection_id
        return connection_id

    def disconnect(self, connection_id: str):
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        # Remove user_id mapping
        for user_id, conn_id in list(self.user_connections.items()):
            if conn_id == connection_id:
                del self.user_connections[user_id]
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

    try:
        if not token:
            await websocket.send_text(json.dumps({"error": "Token is required"}))
            await websocket.close(code=1008)  # Policy Violation
            return

        # Decode and validate the JWT token
        try:
            payload = jwt.decode(token,
                                 os.getenv("AUTH_JWT_SECRET_KEY"),
                                 algorithms=[os.getenv("AUTH_JWT_ALGORITHM")])
            user_id = payload.get("sub")
            user_role = payload.get("role")

            if not user_id:
                await websocket.send_text(json.dumps({"error": "Invalid token"}))
                await websocket.close(code=1008)
                return

        except jwt.ExpiredSignatureError:
            await websocket.send_text(json.dumps({"error": "Token expired"}))
            await websocket.close(code=1008)
            return

        except jwt.InvalidTokenError:
            await websocket.send_text(json.dumps({"error": "Invalid token"}))
            await websocket.close(code=1008)
            return

        # Connect the user
        connection_id = await manager.connect(websocket, user_id)

        # Send welcome message
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "status": "connected",
            "user_id": user_id,
            "role": user_role,
        }))

        # Handle messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)

                # Process message based on type
                message_type = message.get("type", "message")

                if message_type == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))

                elif message_type == "chat":
                    # Process chat message
                    response = await process_chat_message(message.get("content"), user_id)
                    await websocket.send_text(json.dumps({
                        "type": "chat_response",
                        "content": response
                    }))

                else:
                    await websocket.send_text(json.dumps({
                        "type": "echo",
                        "data": message
                    }))

            except WebSocketDisconnect:
                manager.disconnect(connection_id)
                break
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON format"}))

    except WebSocketDisconnect:
        print(f"❌ Websocket disconnected for user_id: {user_id} in locals() else 'Unknown user' ")
    except Exception as e:
        print(f"❌ Websocket error: {str(e)}")
        try:
            await websocket.close(code=1011)  # Internal Error
        except:
            pass

async def process_chat_message(content: str, user_id: str):
    # Placeholder for actual chat processing logic
    # For now, just echo the message back with user_id
    return f"Echo: {content}"