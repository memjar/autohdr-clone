"""
Auth router for Klaus IMI — login, register, user management.
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from imi_auth import (
    add_user,
    authenticate,
    create_token,
    get_current_user,
    load_users,
    require_role,
    JWT_EXPIRY_HOURS,
)

logger = logging.getLogger("imi_auth_router")

router = APIRouter(prefix="/klaus/imi/auth", tags=["imi-auth"])

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    role: str
    expires_at: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str


class UserInfo(BaseModel):
    username: str
    role: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """Authenticate and return a JWT."""
    user = authenticate(req.username, req.password)
    if not user:
        from fastapi import HTTPException

        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_token(user["username"], user["role"])
    from datetime import timedelta

    expires_at = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
    return LoginResponse(token=token, role=user["role"], expires_at=expires_at.isoformat())


@router.post("/register", response_model=UserInfo)
def register(req: RegisterRequest, _user=require_role("admin")):
    """Create a new user (admin only)."""
    try:
        result = add_user(req.username, req.password, req.role)
    except ValueError as e:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=str(e))
    return UserInfo(**result)


@router.get("/me", response_model=UserInfo)
def me(user: dict = Depends(get_current_user)):
    """Return current user info from token."""
    return UserInfo(username=user["username"], role=user["role"])


@router.get("/users")
def list_users(_user=require_role("admin")):
    """List all users without passwords (admin only)."""
    users = load_users()
    return [
        {"username": u["username"], "role": u["role"], "created_at": u.get("created_at")}
        for u in users.values()
    ]
