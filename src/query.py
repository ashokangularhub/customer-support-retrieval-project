import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from retrieval_client import fetch_context
from config import DEFAULT_K, DEFAULT_SEARCH_TYPE

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided context. If the answer isn't in the context, say you don't know."
)

PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])


def ask_question(query: str, collection: str, search_type: str = DEFAULT_SEARCH_TYPE,
                 k: int = DEFAULT_K) -> dict:
    context_chunks = fetch_context(
        query, collection=collection, search_type=search_type, k=k)
    context_text = "\n\n".join(
        f"[{c['doc_type']} | p.{c['page_number']} | {c['content_type']}"
        f"{' | ' + c['table_name'] if c['table_name'] else ''}]\n{c['text']}"
        for c in context_chunks
    )

    messages = PROMPT_TEMPLATE.format_messages(
        context=context_text, question=query)

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    response = llm.invoke(messages)

    return {
        "system_prompt": SYSTEM_PROMPT,
        "context": context_chunks,
        "question": query,
        "search_type": search_type,
        "answer": response.content,
    }


if __name__ == "__main__":
    result = ask_question(
        "What are the trouble shooting steps for AuroraWatch Fit 3?",
        collection="AURORA_TECHNICAL_SUPPORT")
    print(result["answer"])
