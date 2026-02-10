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
