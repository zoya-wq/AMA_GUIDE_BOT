import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Azure Document Intelligence
    DOC_INTELLIGENCE_ENDPOINT = os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT")
    DOC_INTELLIGENCE_KEY = os.getenv("AZURE_DOC_INTELLIGENCE_KEY")
    
    # Azure OpenAI
    OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
    CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
    
    # Cosmos DB
    COSMOS_CONNECTION_STRING = os.getenv("AZURE_COSMOS_CONNECTION_STRING")
    COSMOS_DATABASE = os.getenv("AZURE_COSMOS_DATABASE", "ama_guides")
    COSMOS_CONTAINER = os.getenv("AZURE_COSMOS_CONTAINER", "ama_content")
    
    # Qdrant
    QDRANT_URL = os.getenv("QDRANT_URL", "https://4713750b-3fac-4725-b68c-99f3e17859a8.eu-central-1-0.aws.cloud.qdrant.io:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DlkMuquLnaDPh30tgaeAKrd7eeC-cjAiOd8JRQx4c0o")
    
    # Collections
    QDRANT_COLLECTIONS = {
        "paragraphs": "ama_paragraphs",
        "tables": "ama_tables",
        "formulas": "ama_formulas",
        "sections": "ama_sections"
    }
    
    # Document Intelligence (Azure + LlamaParse)
    USE_LLAMA_PARSE = os.getenv("USE_LLAMA_PARSE", "false").lower() == "true"
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

    # Embedding
    EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", 3072))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
