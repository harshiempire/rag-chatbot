"""
Application configuration.

Loads environment variables and defines application-wide settings.
"""

import os

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

# Database
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost:5432/rag_db')

# CORS
CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173")

# Auth
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET", "change-this-in-production")
REFRESH_TOKEN_PEPPER = os.getenv("REFRESH_TOKEN_PEPPER", ACCESS_TOKEN_SECRET)
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
REFRESH_COOKIE_NAME = os.getenv("REFRESH_COOKIE_NAME", "rag_refresh_token")
AUTH_COOKIE_SECURE = os.getenv("AUTH_COOKIE_SECURE", "false").lower() in {"1", "true", "yes"}
AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "lax")
