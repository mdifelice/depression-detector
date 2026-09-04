import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
WHITELIST_FILE = DATA_DIR / "whitelist.json"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
