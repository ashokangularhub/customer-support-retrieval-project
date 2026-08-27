import importlib
import logging
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

# Make this file's own directory importable regardless of cwd/how it's
# launched, and import local flat modules dynamically (not via `from X
# import Y`) so editor "organize imports" actions can't reorder them above
# this sys.path bootstrap.
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

ask_question = importlib.import_module("query").ask_question
DEFAULT_K = importlib.import_module("config").DEFAULT_K
RequestResponseLoggingMiddleware = importlib.import_module(
    "logging_middleware").RequestResponseLoggingMiddleware

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)

app = FastAPI(title="RAG Retrieval API")
app.add_middleware(RequestResponseLoggingMiddleware)


class QueryRequest(BaseModel):
    question: str
    # Required until a routing agent picks the collection automatically.
    collection: str  # one of AURORA_PRODUCT / AURORA_RETURNS_REFUNDS / AURORA_TECHNICAL_SUPPORT
    search_type: str = "hybrid"  # "keyword", "vector", or "hybrid"
    k: int = DEFAULT_K


class ContextChunk(BaseModel):
    source_file: Optional[str] = None
    doc_type: Optional[str] = None
    page_number: Optional[int] = None
    content_type: Optional[str] = None
    table_name: Optional[str] = None
    section_heading: Optional[str] = None
    product_name: Optional[str] = None
    text: str


class QueryResponse(BaseModel):
    system_prompt: str
    context: List[ContextChunk]
    question: str
    search_type: str
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    result = ask_question(
        request.question, collection=request.collection,
        search_type=request.search_type, k=request.k)
    return QueryResponse(**result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
