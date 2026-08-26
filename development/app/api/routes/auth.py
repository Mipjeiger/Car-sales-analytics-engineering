import os
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, EmailStr
from pathlib import Path
from dotenv import load_dotenv
from typing import Literal
from datetime import datetime, timedelta, timezone
from jose import jwt
import secrets

router = APIRouter()

# Directory setup
BASE_DIR = Path(__file__).resolve().parents[3]
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH)

JWT_SECRET=os.getenv("AUTH_JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("AUTH_JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("AUTH_JWT_EXPIRE_MINUTES", "60"))

USERS = {
    "admin": {
        "email": os.getenv("AUTH_EMAIL_ADMIN", ""),
        "password": os.getenv("AUTH_PASSWORD_ADMIN", ""),
        "role": "admin"
    },
    "analyst": {
        "email": os.getenv("AUTH_EMAIL_ANALYST", ""),
        "password": os.getenv("AUTH_PASSWORD_ANALYST", ""),
        "role": "analyst"
    },
    "viewer": {
        "email": os.getenv("AUTH_EMAIL_VIEWER", ""),
        "password": os.getenv("AUTH_PASSWORD_VIEWER", ""),
        "role": "viewer"
    }
}

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"]
    role: Literal["admin", "analyst", "viewer"]
    email: EmailStr
    expires_in: int

def _find_user(email: str):
    for user in USERS.values():
        if user["email"] and secrets.compare_digest(user["email"], email):
            return user
    return None

@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest):
    user = _find_user(payload.email)
    if not user or not secrets.compare_digest(user["password"], payload.password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    exp = datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    token_payload = {
        "sub": user["email"],
        "role": user["role"],
        "exp": exp
    }
    token = jwt.encode(token_payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return {
        "access_token": token,
        "token_type": "bearer",
        "role": user["role"],
        "email": user["email"],
        "expires_in": JWT_EXPIRE_MINUTES * 60
    }