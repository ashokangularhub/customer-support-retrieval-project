"""HTTP client for the rag-pipeline retrieval API (the "data plane").

rag-retrieval never talks to Weaviate directly — it always goes through
rag-pipeline's /retrieve endpoint, which owns the embedding model and the
Weaviate connection so query-time embeddings always match ingestion-time ones.
"""
import requests

from config import DEFAULT_K, DEFAULT_SEARCH_TYPE, REQUEST_TIMEOUT_SECONDS, RETRIEVE_ENDPOINT


def fetch_context(query: str, collection: str, search_type: str = DEFAULT_SEARCH_TYPE,
                  k: int = DEFAULT_K) -> list[dict]:
    payload = {"question": query, "collection": collection,
               "search_type": search_type, "k": k}
    response = requests.post(
        RETRIEVE_ENDPOINT,
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()["context"]
