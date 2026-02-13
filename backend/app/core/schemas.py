"""
Pydantic schemas for the RAG-Chatbot platform.

All request/response models and configuration schemas used
across the application. Imports enums and secrets from core.
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, validator

from app.core.enums import (
    AuthType,
    ChunkingStrategy,
    DataClassification,
    EmbeddingModel,
    LLMProvider,
    SourceType,
)
from app.core.secrets import SecretField


class AuthConfig(BaseModel):
    """Authentication configuration"""
    auth_type: AuthType = AuthType.NONE
    username: Optional[str] = None
    password: Optional[SecretField] = None
    api_key: Optional[SecretField] = None
    bearer_token: Optional[SecretField] = None
    client_id: Optional[str] = None
    client_secret: Optional[SecretField] = None
    token_url: Optional[str] = None

    def encrypt_secrets(self):
        for field in [self.password, self.api_key, self.bearer_token, self.client_secret]:
            if field:
                field.encrypt_value()


class ConnectionConfig(BaseModel):
    """Connection configuration for any source type"""
    # API
    url: Optional[str] = None
    method: Optional[str] = "GET"
    headers: Optional[Dict[str, str]] = None
    auth: Optional[AuthConfig] = None

    # Database
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[SecretField] = None

    # eCFR
    title_number: Optional[int] = None
    date: Optional[str] = None

    # File
    file_path: Optional[str] = None

    def encrypt_all_secrets(self):
        if self.password:
            self.password.encrypt_value()
        if self.auth:
            self.auth.encrypt_secrets()


class DataSourceDefinition(BaseModel):
    """Complete data source definition"""
    id: Optional[str] = None
    name: str
    type: SourceType
    config: ConnectionConfig
    classification: DataClassification = DataClassification.PUBLIC
    created_at: Optional[datetime] = None

    @validator('id', pre=True, always=True)
    def set_id(cls, v):
        return v or str(uuid.uuid4())

    @validator('created_at', pre=True, always=True)
    def set_created_at(cls, v):
        return v or datetime.now()

    def encrypt_secrets(self):
        self.config.encrypt_all_secrets()


class ChunkingConfig(BaseModel):
    """Text chunking configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 200


class VectorizationConfig(BaseModel):
    """Vectorization configuration"""
    embedding_model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS
    batch_size: int = 100
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.AUTO


class ReviewDecision(BaseModel):
    """Document review decision"""
    document_id: str
    reviewer_id: str
    decision: Literal["approve", "reject"]
    comments: Optional[str] = None
    classification_override: Optional[DataClassification] = None


class RAGQuery(BaseModel):
    """RAG query request"""
    question: str
    llm_provider: LLMProvider
    classification_filter: Optional[List[DataClassification]] = None
    source_id: Optional[str] = None
    top_k: int = 5
    temperature: float = 0.7
    min_similarity: float = 0.2
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None


class RAGResponse(BaseModel):
    """RAG query response"""
    answer: str
    sources: List[Dict[str, Any]]
    classification: DataClassification
    llm_provider: LLMProvider
    retrieved_count: int
    prompt_context_count: int
    total_ms: float
    timings_ms: Dict[str, float]


class UserSignupRequest(BaseModel):
    """User registration payload."""

    email: str
    password: str = Field(min_length=8, max_length=128)

    @validator("email")
    def normalize_email_value(cls, value: str):
        normalized = value.strip().lower()
        if "@" not in normalized:
            raise ValueError("Email must be valid.")
        return normalized


class UserLoginRequest(BaseModel):
    """User login payload."""

    email: str
    password: str = Field(min_length=1, max_length=128)

    @validator("email")
    def normalize_email_value(cls, value: str):
        normalized = value.strip().lower()
        if "@" not in normalized:
            raise ValueError("Email must be valid.")
        return normalized


class UserPublic(BaseModel):
    """Public user payload."""

    id: str
    email: str
    created_at: datetime


class AuthTokenResponse(BaseModel):
    """Access-token response payload."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserPublic


class AuthMessageResponse(BaseModel):
    """Simple auth response message."""

    message: str


class ChatHistoryTurn(BaseModel):
    """Conversation turn used for context."""

    role: Literal["user", "assistant"]
    content: str


class ChatSessionPayload(BaseModel):
    """Persisted chat session payload."""

    id: str
    title: str
    llmProvider: LLMProvider
    createdAt: int
    updatedAt: int
    messages: List[Dict[str, Any]] = Field(default_factory=list)
