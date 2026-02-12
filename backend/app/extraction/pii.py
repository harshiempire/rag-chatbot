"""
PII (Personally Identifiable Information) detector.

Scans text for common PII patterns: emails, SSNs, phone numbers,
and credit card numbers using regular expressions.
"""

import re
from typing import Dict


class PIIDetector:
    """Detect personally identifiable information"""

    def __init__(self):
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b'
        }

    def detect(self, text: str) -> bool:
        """Detect if text contains PII"""
        for pattern in self.patterns.values():
            if re.search(pattern, text):
                return True
        return False

    def get_pii_details(self, text: str) -> Dict:
        """Get detailed PII information"""
        detected = {}
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = len(matches)

        return {
            'has_pii': bool(detected),
            'types': detected,
            'count': sum(detected.values())
        }
