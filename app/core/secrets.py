"""
Secret management for the RAG-Chatbot platform.

Handles encryption/decryption of sensitive credentials (API keys,
passwords, tokens) using Fernet symmetric encryption.
"""

import base64
import os
from typing import Optional

from cryptography.fernet import Fernet
from pydantic import BaseModel


class SecretManager:
    """Handles encryption/decryption of secrets"""

    def __init__(self):
        key = os.getenv('ENCRYPTION_KEY')
        if not key:
            key = Fernet.generate_key().decode()
            print(f"WARNING: ENCRYPTION_KEY not set. Generated: {key}")
            print("WARNING: Set as environment variable in production!")
        self.cipher = Fernet(key.encode() if isinstance(key, str) else key)

    def encrypt(self, plaintext: str) -> str:
        if not plaintext:
            return ""
        encrypted = self.cipher.encrypt(plaintext.encode())
        return base64.b64encode(encrypted).decode()

    def decrypt(self, encrypted_text: str) -> str:
        if not encrypted_text:
            return ""
        decoded = base64.b64decode(encrypted_text.encode())
        return self.cipher.decrypt(decoded).decode()

    def mask_secret(self, secret: str, visible_chars: int = 4) -> str:
        if not secret or len(secret) <= visible_chars:
            return "****"
        return f"{'*' * (len(secret) - visible_chars)}{secret[-visible_chars:]}"


# Module-level singleton
secret_manager = SecretManager()


class SecretField(BaseModel):
    """Encrypted secret field"""
    name: str
    value: str  # Will be encrypted
    masked_value: Optional[str] = None

    def encrypt_value(self):
        if self.value and not self.value.startswith("enc:"):
            encrypted = secret_manager.encrypt(self.value)
            self.masked_value = secret_manager.mask_secret(self.value)
            self.value = f"enc:{encrypted}"

    def get_decrypted_value(self) -> str:
        if self.value.startswith("enc:"):
            return secret_manager.decrypt(self.value.replace("enc:", ""))
        return self.value
