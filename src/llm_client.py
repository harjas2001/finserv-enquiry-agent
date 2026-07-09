"""
src/llm_client.py
==================
Centralised Vertex AI (agent platform technically) client factory for the whole application.
 
Why this file exists:
    Before the Vertex AI migration, five separate files (orchestrator.py,
    account.py, product.py, complaint.py, embeddings.py) each read
    GOOGLE_API_KEY from the environment and built their own genai.Client.
    Auth logic lived in five places, one change meant five edits.
 
    This factory is the single source of truth for how the app authenticates
    to Gemini. Every subagent and the embeddings wrapper import get_client()
    instead of constructing their own client.
 
Auth model (Vertex AI, not AI Studio):
    No API key anywhere in the codebase. Authentication is via Application
    Default Credentials (ADC):
      - Locally: `gcloud auth application-default login` writes a credentials
        file that the client library discovers automatically.
      - On Cloud Run: the service account attached to the revision is used
        automatically — no credentials file, no secret, no key material in
        the container image or the environment.
 
Singleton:
    get_client() caches one Client instance at module level and reuses it on
    every call. Building a genai.Client sets up an underlying HTTP client;
    creating a fresh one per graph node invocation (the old per-call pattern)
    is wasted setup cost on every request. A container instance now builds
    its client once and reuses it for the life of the instance.
"""

import os 
from google import genai

_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT")
_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "australia-southeast1")
_client = genai.Client | None=None


def get_client() -> genai.Client:
    """
    Returns a shared Vertex AI-backed genai.Client, creating it on first call.
 
    Raises EnvironmentError immediately if GOOGLE_CLOUD_PROJECT isn't set, so
    a misconfigured deploy fails fast and loud instead of surfacing as a
    cryptic 401/403 three layers down inside a subagent.
    """
    global _client

    if _client is None:
        if not _PROJECT:
            raise EnvironmentError(
                "GOOGLE_CLOUD_PROJECT is not set. Required for Agent Platform auth."
            )
        
        _client = genai.Client(
            vertexai=True,
            project=_PROJECT,
            location=_LOCATION,
        )

    return _client