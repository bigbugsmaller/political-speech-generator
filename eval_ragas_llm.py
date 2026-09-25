"""RAGAS LLM/embeddings wired to the same OpenAI-compatible config as the speech agent."""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import MODEL, MODEL_URL, OPENAI_API
from ragas.embeddings import HuggingFaceEmbeddings, LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

_EMBED_MODEL = "text-embedding-3-small"
_HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_ragas_llm() -> LangchainLLMWrapper:
    chat = ChatOpenAI(
        model=MODEL,
        api_key=OPENAI_API,
        base_url=MODEL_URL,
        temperature=0.0,
    )
    return LangchainLLMWrapper(chat)


def build_ragas_embeddings():
    """Answer relevancy needs embeddings; prefer OpenRouter-compatible API, else local HF."""
    try:
        emb = OpenAIEmbeddings(
            model=_EMBED_MODEL,
            api_key=OPENAI_API,
            base_url=MODEL_URL,
        )
        # Probe once so we can fall back before the full eval batch.
        emb.embed_query("ping")
        return LangchainEmbeddingsWrapper(emb)
    except Exception:
        return HuggingFaceEmbeddings(model=_HF_EMBED_MODEL, interface="modern")
