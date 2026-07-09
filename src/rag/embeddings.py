"""
src/rag/embeddings.py
=====================
Custom LangChain-compatible embeddings class using the google-genai SDK directly.

Why not langchain-google-genai's built-in embeddings?
  The langchain-google-genai wrapper internally routes to v1beta and does not
  expose the current gemini-embedding-* models for all API key tiers.
  This class uses the google-genai client directly for explicit control over
  the API version and model name.

Model: gemini-embedding-2
  The current stable Google embedding model. The batch embedContent API
  returns inconsistent results with this model — embed_documents calls
  the API once per text to guarantee len(output) == len(input), which is
  what ChromaDB requires. 15 sequential calls at ingest time is negligible.

API version: v1
  The SDK defaults to v1beta. Forcing v1 is best practice for production
  code and avoids routing issues with newer model names.
"""

from typing import List
from langchain_core.embeddings import Embeddings

from src.llm_client import get_client


class GeminiEmbeddings(Embeddings):
    """
    LangChain-compatible embeddings backed by Google's gemini-embedding-2 model.

    Implements the two methods LangChain/ChromaDB requires:
      embed_documents(texts)  → list of float vectors (one per document chunk)
      embed_query(text)       → single float vector (for retrieval queries)

    embed_documents calls the API once per text (not batched) because
    gemini-embedding-2's batch behaviour returns inconsistent result counts.
    At 15 chunks this is fast enough; batch behaviour is not worth the fragility.
    """

    def __init__(self, model: str = "gemini-embedding-2"):
        self.model = model
        self.client = get_client()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document chunks, one API call per text.

        Why one at a time:
          gemini-embedding-2's batch embedContent returns unpredictable result
          counts — ChromaDB raises IndexError when len(embeddings) != len(texts).
          Sequential calls guarantee a 1:1 mapping and are reliable for 15 chunks.
        """
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=[text],
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string. Called by ChromaDB at retrieval time."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=[text],
        )
        return response.embeddings[0].values