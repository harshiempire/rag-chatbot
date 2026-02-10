"""
Dynamic data extractor for multiple source types.

Routes extraction to the appropriate handler based on source type
(eCFR, REST API, PostgreSQL/MySQL databases).
"""

import logging
from typing import Dict, List

import pandas as pd
import requests

from app.extraction.ecfr import eCFRExtractor

logger = logging.getLogger(__name__)


class DynamicExtractor:
    """Extract from any configured source"""

    def __init__(self, source_definition):
        self.source_def = source_definition

    def extract(self) -> pd.DataFrame:
        """Extract data based on source type"""

        if self.source_def.type.value == 'ecfr_api':
            return self._extract_ecfr()

        elif self.source_def.type.value == 'rest_api':
            return self._extract_rest_api()

        elif self.source_def.type.value in ['postgres', 'mysql']:
            return self._extract_database()

        else:
            raise ValueError(f"Unsupported source type: {self.source_def.type}")

    def _extract_ecfr(self) -> pd.DataFrame:
        """Extract from eCFR"""
        extractor = eCFRExtractor(
            title_number=self.source_def.config.title_number,
            date=self.source_def.config.date,
            part=getattr(self.source_def.config, "part", None),
            content_mode=getattr(self.source_def.config, "content_mode", "full")
        )
        docs = extractor.extract()
        return pd.DataFrame(docs)

    def _extract_rest_api(self) -> pd.DataFrame:
        """Extract from REST API"""
        config = self.source_def.config
        headers = config.headers or {}

        # Add authentication
        if config.auth:
            auth = config.auth
            if auth.auth_type.value == 'bearer_token' and auth.bearer_token:
                headers['Authorization'] = f"Bearer {auth.bearer_token.get_decrypted_value()}"
            elif auth.auth_type.value == 'api_key' and auth.api_key:
                headers['X-API-Key'] = auth.api_key.get_decrypted_value()
            elif auth.auth_type.value == 'basic' and auth.username and auth.password:
                import base64
                credentials = f"{auth.username}:{auth.password.get_decrypted_value()}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers['Authorization'] = f"Basic {encoded}"

        # Make request
        response = requests.request(
            method=config.method,
            url=config.url,
            headers=headers
        )
        response.raise_for_status()

        data = response.json()

        # Convert to DataFrame
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict) and 'results' in data:
            return pd.DataFrame(data['results'])
        else:
            return pd.DataFrame([data])

    def _extract_database(self) -> pd.DataFrame:
        """Extract from database"""
        import psycopg2

        config = self.source_def.config

        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.username,
            password=config.password.get_decrypted_value() if config.password else None
        )

        query = "SELECT * FROM your_table LIMIT 1000"  # Customize as needed
        df = pd.read_sql(query, conn)
        conn.close()

        return df
