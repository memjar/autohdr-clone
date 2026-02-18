"""
Role-Based Access Control for Klaus IMI Platform.
JWT auth with 3 roles: admin, analyst, viewer.
"""

import hashlib
import json
import logging
import os
import secrets
import time

from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Request

logger = logging.getLogger("imi_auth")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JWT_SECRET = os.environ.get("IMI_JWT_SECRET", "klaus-imi-dev-secret-2026")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 24

VALID_ROLES = {"admin", "analyst", "viewer"}

USERS_DIR = os.path.expanduser("~/.axe/klaus_auth")
USERS_FILE = os.path.join(USERS_DIR, "users.json")

# ---------------------------------------------------------------------------
# Password hashing (sha256 + salt)
# ---------------------------------------------------------------------------


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Return (hash_hex, salt_hex)."""
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return h, salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    h, _ = _hash_password(password, salt)
    return h == stored_hash


# ---------------------------------------------------------------------------
# User storage
# ---------------------------------------------------------------------------


def load_users() -> dict:
    """Load users dict from disk. Keys are usernames."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load users: %s", e)
        return {}


def save_users(users: dict):
    """Persist users dict to disk."""
    os.makedirs(USERS_DIR, exist_ok=True)
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def add_user(username: str, password: str, role: str) -> dict:
    """Add a user. Returns the user record (without password)."""
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role}. Must be one of {VALID_ROLES}")
    users = load_users()
    if username in users:
        raise ValueError(f"User '{username}' already exists")
    pw_hash, salt = _hash_password(password)
    users[username] = {
        "username": username,
        "role": role,
        "password_hash": pw_hash,
        "salt": salt,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    save_users(users)
    return {"username": username, "role": role}


def authenticate(username: str, password: str) -> dict | None:
    """Verify credentials. Returns {username, role} or None."""
    users = load_users()
    user = users.get(username)
    if not user:
        return None
    if not _verify_password(password, user["password_hash"], user["salt"]):
        return None
    return {"username": user["username"], "role": user["role"]}


# ---------------------------------------------------------------------------
# Seed defaults
# ---------------------------------------------------------------------------


def _seed_defaults():
    users = load_users()
    changed = False
    for uname, pw, role in [
        ("admin", "klaus2026", "admin"),
        ("analyst", "analyst2026", "analyst"),
    ]:
        if uname not in users:
            pw_hash, salt = _hash_password(pw)
            users[uname] = {
                "username": uname,
                "role": role,
                "password_hash": pw_hash,
                "salt": salt,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            changed = True
            logger.info("Seeded default user: %s (%s)", uname, role)
    if changed:
        save_users(users)


_seed_defaults()

# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_token(username: str, role: str) -> str:
    """Create a JWT with 24h expiry."""
    exp = datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
    payload = {
        "sub": username,
        "role": role,
        "exp": exp,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> dict:
    """Decode and verify a JWT. Returns {username, role, exp}."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return {
            "username": payload["sub"],
            "role": payload["role"],
            "exp": payload["exp"],
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------


def _extract_token(request: Request) -> str:
    """Pull Bearer token from Authorization header."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return auth[7:]


def get_current_user(request: Request) -> dict:
    """FastAPI dependency: returns {username, role, exp} from JWT."""
    token = _extract_token(request)
    return verify_token(token)


def require_role(*roles: str):
    """FastAPI dependency factory: require the caller to have one of the given roles."""

    def dependency(request: Request) -> dict:
        user = get_current_user(request)
        if user["role"] not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{user['role']}' not permitted. Required: {list(roles)}",
            )
        return user

    return Depends(dependency)
