import os
import sys
from dotenv import load_dotenv

load_dotenv()

from openai import AzureOpenAI

def test_azure_openai():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    key = os.getenv("AZURE_OPENAI_KEY")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-5.1-mini")

    print("Checking keys:")
    print(f"Endpoint: {endpoint}")
    print(f"Key format looks ok (len={len(key) if key else 0})")
    print(f"Embedding model: {embedding_deployment}")
    print(f"Chat model: {chat_deployment}")

    if not endpoint or not key:
        print("Missing endpoint or key!")
        return

    try:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=key,
            api_version = "2024-12-01-preview"
        )
    except Exception as e:
        print(f"Failed to init client: {e}")
        return

    print("\n1. Testing Embedding Model...")
    try:
        response = client.embeddings.create(
            model=embedding_deployment,
            input="Test sentence for embedding"
        )
        print(f"Embedding success! Vector length: {len(response.data[0].embedding)}")
    except Exception as e:
        print(f"Embedding failed: {e}")

    print("\n2. Testing Chat Model...")
    try:
        response = client.chat.completions.create(
            model=chat_deployment,
            messages=[{"role": "user", "content": "Say 'hello world' if you hear me. Be very brief."}],
            max_completion_tokens=10 # Updated parameter
        )
        print(f"Chat success! Response: '{response.choices[0].message.content}'")
    except Exception as e:
        print(f"Chat failed: {e}")

if __name__ == "__main__":
    test_azure_openai()
