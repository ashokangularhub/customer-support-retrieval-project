import os

RAG_PIPELINE_URL = os.getenv("RAG_PIPELINE_URL", "http://localhost:8000")
RETRIEVE_ENDPOINT = f"{RAG_PIPELINE_URL}/retrieve"

DEFAULT_SEARCH_TYPE = "hybrid"
DEFAULT_K = int(os.getenv("RAG_TOP_K", "4"))
REQUEST_TIMEOUT_SECONDS = 30
