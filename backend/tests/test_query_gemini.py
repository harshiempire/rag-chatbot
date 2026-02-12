"""
Test RAG query with Google Gemini API
"""
import requests
import json

url = "http://localhost:8000/api/v1/rag/query"

payload = {
    "question": "What are the capital requirements for banks under Title 12?",
    "llm_provider": "google",  # Use Gemini
    "classification_filter": ["public"],
    "top_k": 3,
    "temperature": 0.7,
    "min_similarity": 0.2
}

print("🚀 Testing RAG query with Google Gemini...")
print(f"📝 Question: {payload['question']}")
print("-" * 50)

try:
    response = requests.post(url, json=payload, timeout=60)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Answer:\n{result['answer']}")
        print(f"\n📚 Sources used: {len(result['sources'])}")
        for i, source in enumerate(result['sources'][:3]):
            print(f"  [{i+1}] Similarity: {source.get('similarity', 'N/A'):.2f}")
    else:
        print(f"❌ Error: {response.json()}")
        
except requests.exceptions.Timeout:
    print("❌ ERROR: Request timed out after 60s")
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Cannot connect to API. Is FastAPI running on port 8000?")
    print("   Run: uvicorn part4_rag_api:app --reload --port 8000")
except Exception as e:
    print(f"❌ ERROR: {e}")
