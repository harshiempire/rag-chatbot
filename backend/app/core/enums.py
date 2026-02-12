"""
Enumerations for the RAG-Chatbot platform.

Defines all enum types used across the application:
source types, authentication types, data classifications,
document statuses, LLM providers, and embedding models.
"""

from enum import Enum


class SourceType(str, Enum):
    POSTGRES = "postgres"
    MYSQL = "mysql"
    REST_API = "rest_api"
    ECFR_API = "ecfr_api"
    CSV = "csv"
    JSON = "json"


class AuthType(str, Enum):
    NONE = "none"
    BASIC = "basic"
    BEARER_TOKEN = "bearer_token"
    API_KEY = "api_key"
    OAUTH2_CLIENT_CREDENTIALS = "oauth2_client_credentials"


class DataClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DocumentStatus(str, Enum):
    PENDING_EXTRACTION = "pending_extraction"
    EXTRACTED = "extracted"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    VECTORIZED = "vectorized"
    PUBLISHED = "published"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class EmbeddingModel(str, Enum):
    OPENAI_ADA_002 = "text-embedding-ada-002"
    OPENAI_3_SMALL = "text-embedding-3-small"
    SENTENCE_TRANSFORMERS = "all-MiniLM-L6-v2"


class ChunkingStrategy(str, Enum):
    AUTO = "auto"
    FIXED = "fixed"
    STRUCTURE_AWARE = "structure_aware"
