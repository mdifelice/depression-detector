import json
from pathlib import Path
from fastapi import Request, HTTPException
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from config import (
    GOOGLE_CLIENT_ID,
    WHITELIST_FILE,
    SECRET_KEY,
)
from models import User


def load_whitelist() -> list[str]:
    if not WHITELIST_FILE.exists():
        return []
    with open(WHITELIST_FILE, "r") as f:
        return json.load(f)


def save_whitelist(emails: list[str]):
    WHITELIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WHITELIST_FILE, "w") as f:
        json.dump(emails, f, indent=2)


def is_whitelisted(email: str) -> bool:
    return email.lower() in [e.lower() for e in load_whitelist()]


def verify_google_token(token: str) -> dict:
    try:
        idinfo = id_token.verify_oauth2_token(
            token, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        return idinfo
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google token")


def get_current_user(request: Request) -> User:
    session = request.session
    if "user" not in session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return User(**session["user"])
