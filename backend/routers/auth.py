from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REDIRECT_URI,
    FRONTEND_URL,
    DISABLE_AUTH,
)
from auth import is_whitelisted, verify_google_token
from models import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/login")
async def login():
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
    )
    return RedirectResponse(url=authorization_url)


@router.get("/callback")
async def callback(request: Request):
    code = request.query_params.get("code")
    if not code:
        raise HTTPException(status_code=400, detail="Missing authorization code")

    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": GOOGLE_CLIENT_ID,
                "client_secret": GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    )
    flow.redirect_uri = GOOGLE_REDIRECT_URI

    flow.fetch_token(code=code)
    credentials = flow.credentials

    idinfo = id_token.verify_oauth2_token(
        credentials.id_token, google_requests.Request(), GOOGLE_CLIENT_ID
    )

    email = idinfo.get("email", "")
    if not is_whitelisted(email):
        raise HTTPException(status_code=403, detail="Email not whitelisted. Contact administrator.")

    user = User(
        email=email,
        name=idinfo.get("name", ""),
        picture=idinfo.get("picture", ""),
    )

    request.session["user"] = user.model_dump()
    return RedirectResponse(url=f"{FRONTEND_URL}/dashboard")


@router.post("/login-token")
async def login_with_token(request: Request):
    body = await request.json()
    token = body.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="Missing token")

    idinfo = verify_google_token(token)
    email = idinfo.get("email", "")

    if not is_whitelisted(email):
        raise HTTPException(status_code=403, detail="Email not whitelisted. Contact administrator.")

    user = User(
        email=email,
        name=idinfo.get("name", ""),
        picture=idinfo.get("picture", ""),
    )

    request.session["user"] = user.model_dump()
    return {"user": user.model_dump()}


@router.post("/logout")
async def logout(request: Request):
    request.session.clear()
    return {"message": "Logged out"}


@router.get("/me")
async def me(request: Request):
    if DISABLE_AUTH:
        return {"email": "test@localhost", "name": "Test User", "picture": ""}
    if "user" not in request.session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.session["user"]
