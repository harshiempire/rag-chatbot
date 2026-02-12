"""
Authentication helpers for password hashing and token handling.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Tuple

from app.config import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    ACCESS_TOKEN_SECRET,
    REFRESH_TOKEN_EXPIRE_DAYS,
    REFRESH_TOKEN_PEPPER,
)

PBKDF2_ALGORITHM = "pbkdf2_sha256"
PBKDF2_ITERATIONS = 260_000
PBKDF2_SALT_BYTES = 16


class AuthError(ValueError):
    """Raised when an auth token or credential is invalid."""


def _to_b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _from_b64url(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_email(email: str) -> str:
    return email.strip().lower()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(PBKDF2_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PBKDF2_ITERATIONS,
    )
    return f"{PBKDF2_ALGORITHM}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algorithm, iterations_raw, salt_hex, digest_hex = stored_hash.split("$", 3)
        if algorithm != PBKDF2_ALGORITHM:
            return False
        iterations = int(iterations_raw)
        expected = bytes.fromhex(digest_hex)
        salt = bytes.fromhex(salt_hex)
    except (ValueError, TypeError):
        return False

    computed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(computed, expected)


def create_access_token(user_id: str, email: str) -> Tuple[str, int]:
    now = _utcnow()
    expires_in = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    payload = {
        "sub": user_id,
        "email": email,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
    }
    header = {"alg": "HS256", "typ": "JWT"}

    header_segment = _to_b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_segment = _to_b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    signature = hmac.new(ACCESS_TOKEN_SECRET.encode("utf-8"), signing_input, hashlib.sha256).digest()
    token = f"{header_segment}.{payload_segment}.{_to_b64url(signature)}"
    return token, expires_in


def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        header_segment, payload_segment, signature_segment = token.split(".", 2)
    except ValueError as exc:
        raise AuthError("Invalid token format.") from exc

    signing_input = f"{header_segment}.{payload_segment}".encode("ascii")
    expected_signature = hmac.new(
        ACCESS_TOKEN_SECRET.encode("utf-8"),
        signing_input,
        hashlib.sha256,
    ).digest()

    try:
        provided_signature = _from_b64url(signature_segment)
    except Exception as exc:
        raise AuthError("Invalid token signature.") from exc

    if not hmac.compare_digest(expected_signature, provided_signature):
        raise AuthError("Invalid token signature.")

    try:
        payload = json.loads(_from_b64url(payload_segment))
    except Exception as exc:
        raise AuthError("Invalid token payload.") from exc

    token_type = payload.get("type")
    exp = payload.get("exp")
    user_id = payload.get("sub")
    email = payload.get("email")
    if token_type != "access" or not isinstance(exp, int) or not user_id or not email:
        raise AuthError("Invalid token claims.")
    if exp < int(_utcnow().timestamp()):
        raise AuthError("Token has expired.")
    return payload


def issue_refresh_token() -> Tuple[str, str, datetime]:
    raw_token = secrets.token_urlsafe(48)
    token_hash = hash_refresh_token(raw_token)
    expires_at = _utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    return raw_token, token_hash, expires_at


def hash_refresh_token(raw_token: str) -> str:
    return hashlib.sha256(f"{raw_token}{REFRESH_TOKEN_PEPPER}".encode("utf-8")).hexdigest()
