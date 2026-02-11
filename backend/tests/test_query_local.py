import requests
import json

url = "http://localhost:8000/api/v1/rag/query"

payload = {
    "question": "What are the capital requirements for banks?",
    "llm_provider": "local",
    "classification_filter": ["public"],
    "top_k": 3,
    "temperature": 0.7,
    "min_similarity": 0.2
}

print("Testing RAG query...")
try:
    response = requests.post(url, json=payload, timeout=35)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except requests.exceptions.Timeout:
    print("ERROR: Request timed out after 35s")
except requests.exceptions.ConnectionError:
    print("ERROR: Cannot connect to API. Is FastAPI running on port 8000?")
except Exception as e:
    print(f"ERROR: {e}")
    if hasattr(e, 'response'):
        print(f"Response: {e.response.text}")
