import requests
import json

url = "http://localhost:8000/api/v1/rag/query"

payload = {
    "question": "What are the capital requirements for banks?",
    "llm_provider": "openai",  # Changed from "local"
    "classification_filter": ["public"],
    "top_k": 3,
    "temperature": 0.7,
    "min_similarity": 0.2
}

print("Testing RAG query with OpenAI...")
try:
    response = requests.post(url, json=payload, timeout=35)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"ERROR: {e}")
