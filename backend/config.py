import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR.parent / "data")))
MODELS_DIR = Path(os.getenv("MODELS_DIR", str(BASE_DIR.parent / "models")))
WHITELIST_FILE = DATA_DIR / "whitelist.json"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")

SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-production")
SESSION_MAX_AGE = 60 * 60 * 24 * 7  # 7 days

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
DISABLE_AUTH = os.getenv("DISABLE_AUTH", "false").lower() == "true"

ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
