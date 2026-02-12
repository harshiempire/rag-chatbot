"""
eCFR (Electronic Code of Federal Regulations) extractor.

Extracts regulatory text from the eCFR API, supporting both
structure-only and full XML (with text content) extraction modes.
"""

import logging
import xml.etree.ElementTree as ET
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)


class eCFRExtractor:
    """Extract regulations from eCFR API"""

    def __init__(self, title_number: int, date: str = None, part: str = None, content_mode: str = "full"):
        self.title_number = title_number
        self.date = date or "current"
        self.part = part
        self.content_mode = content_mode
        self.base_url = "https://www.ecfr.gov/api/versioner/v1"

    def extract(self) -> List[Dict]:
        """Extract regulations"""
        if self.content_mode == "structure":
            return self._extract_structure()
        if self.content_mode == "full":
            return self._extract_full_xml()
        raise ValueError(f"Unsupported content mode: {self.content_mode}")

    def _extract_structure(self) -> List[Dict]:
        """Extract structure only (no text content)"""
        url = f"{self.base_url}/structure/{self.date}/title-{self.title_number}.json"
        logger.info(f"Extracting eCFR structure: Title {self.title_number}")

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        documents = []
        self._traverse(data, documents, [])

        logger.info(f"Extracted {len(documents)} structure nodes")
        return documents

    def _extract_full_xml(self) -> List[Dict]:
        """Extract full text using eCFR full XML endpoint"""
        url = f"{self.base_url}/full/{self.date}/title-{self.title_number}.xml"
        if self.part:
            url = f"{url}?part={self.part}"

        logger.info(
            f"Extracting eCFR full text: Title {self.title_number}, Part {self.part or 'ALL'}"
        )

        response = requests.get(url)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        documents = []

        for div in root.iter():
            if div.tag.startswith("DIV") and div.attrib.get("TYPE") == "SECTION":
                head = div.find("HEAD")
                text_parts = []
                if head is not None and head.text:
                    text_parts.append(head.text.strip())

                for p in div.findall("P"):
                    p_text = "".join(p.itertext()).strip()
                    if p_text:
                        text_parts.append(p_text)

                content = "\n".join(text_parts).strip()
                if content and len(content) > 100:
                    documents.append({
                        "content": content,
                        "metadata": {
                            "title": self.title_number,
                            "part": self.part,
                            "section": div.attrib.get("N", ""),
                            "heading": head.text.strip() if head is not None and head.text else "",
                            "type": div.attrib.get("TYPE", ""),
                            "source": "ecfr",
                            "date": self.date
                        }
                    })

        logger.info(f"Extracted {len(documents)} sections with text")
        return documents

    def _traverse(self, node, documents, path):
        """Recursively traverse eCFR structure"""
        if isinstance(node, dict):
            content = node.get('text', '')

            # Only store meaningful content
            if content and len(content) > 100:
                documents.append({
                    'content': content,
                    'metadata': {
                        'title': self.title_number,
                        'path': ' > '.join(path),
                        'identifier': node.get('identifier', ''),
                        'label': node.get('label', ''),
                        'type': node.get('type', ''),
                        'source': 'ecfr'
                    }
                })

            # Traverse children
            if 'children' in node:
                for child in node['children']:
                    self._traverse(child, documents, path + [node.get('label', '')])
