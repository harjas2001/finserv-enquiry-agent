"""
Phase 2 — RAG Retriever
src/rag/retriever.py

Public API:
    retrieve(query: str) -> list[dict]
    format_context(results) -> str

The retriever is called by the product_information subagent.
The ingestion pipeline (ingest.py) must be run first to populate ChromaDB.

Run directly to test:
    python -m src.rag.retriever
"""

import os
import sys
from pathlib import Path

from src.rag.embeddings import GeminiEmbeddings
from langchain_chroma import Chroma

# ── Configuration ─────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME    = "product_knowledge"
TOP_K              = 3   # How many chunks to retrieve per query


# ── Core retriever factory ────────────────────────────────────────────────────

def get_vectorstore() -> Chroma:
    """
    Load the persisted ChromaDB from disk.

    This does NOT re-embed anything — it just connects to the existing database
    that ingest.py created. It does need the embedding model to embed queries
    at search time.
    """
    embeddings = GeminiEmbeddings(model="gemini-embedding-2")

    if not Path(CHROMA_PERSIST_DIR).exists():
        raise FileNotFoundError(
            f"ChromaDB not found at '{CHROMA_PERSIST_DIR}'. "
            "Run  python -m src.rag.ingest  first."
        )

    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    return vectorstore


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """
    Retrieve the top-K most semantically relevant chunks for a query.

    What happens under the hood:
    1. The query string is embedded → 3072-dim vector (one API call)
    2. ChromaDB computes cosine similarity between the query vector
       and every stored chunk vector
    3. The top-K chunks with highest similarity are returned
    4. We extract text + source metadata and return as plain dicts

    Returns:
        List of dicts, each with keys:
            "text"   — the chunk text (to inject into the LLM prompt)
            "source" — the filename the chunk came from
    """
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)

    results = []
    for doc in docs:
        results.append({
            "text":   doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
        })

    return results


def retrieve_with_scores(query: str, k: int = TOP_K) -> list[dict]:
    """
    Same as retrieve() but also returns the similarity score.

    Useful during development to understand how well retrieval is working.
    Scores are cosine similarity (0.0–1.0). Above 0.75 is generally good.
    Below 0.50 means the query likely has no good match in the knowledge base.
    """
    vectorstore = get_vectorstore()
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)

    results = []
    for doc, score in docs_and_scores:
        results.append({
            "text":   doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "score":  round(float(score), 4),
        })

    return results


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a single string for prompt injection.

    Example output:
        [Source 1: home_loan_guide.txt]
        Variable Rate: 6.54% p.a. ...

        ---

        [Source 2: borrowing_guide.txt]
        The comparison rate includes fees ...

    The product subagent's prompt instructs: "Answer ONLY from the context below."
    This format makes it clear to the LLM which text is sourced from the KB,
    and allows it to cite sources if needed.
    """
    if not results:
        return "No relevant information found in the knowledge base."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[Source {i}: {r['source']}]\n{r['text']}")

    return "\n\n---\n\n".join(parts)


# ── Test harness ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("RAG RETRIEVER TEST HARNESS")
    print("=" * 60)

    # These queries test different parts of the knowledge base.
    # For each, we expect retrieval from a specific document:
    #   Query 1 → home_loan_guide.txt       (variable rate)
    #   Query 2 → personal_loan_facts.txt   (extra repayments)
    #   Query 3 → savings_account_summary.txt (bonus rate)
    #   Query 4 → borrowing_guide.txt       (offset definition)
    #   Query 5 → should return low-relevance results (nothing in KB about this)

    test_cases = [
        ("What is the variable interest rate on a home loan?",   "home_loan_guide.pdf"),
        ("Can I make extra repayments on my personal loan?",     "personal_loan_facts.pdf"),
        ("What are the conditions to earn the bonus interest?",  "savings_account_summary.pdf"),
        ("How does an offset account work?",                     "borrowing_guide.pdf"),
        ("What is the exchange rate for US dollars?",            "N/A — out of scope"),
    ]

    all_passed = True

    for query, expected_source in test_cases:
        print(f"\nQuery: '{query}'")
        print(f"Expected primary source: {expected_source}")
        print("-" * 50)

        results = retrieve_with_scores(query)

        for i, r in enumerate(results, 1):
            snippet = r["text"][:120].replace("\n", " ").strip()
            print(f"  [{i}] score={r['score']:.4f}  source={r['source']}")
            print(f"       {snippet}...")

        # Check if expected source is in top result
        if expected_source != "N/A — out of scope":
            top_source = results[0]["source"] if results else ""
            passed = expected_source in top_source
            status = "✓ PASS" if passed else "✗ FAIL"
            if not passed:
                all_passed = False
            print(f"\n  Routing check: {status}")

    print("\n" + "=" * 60)
    print(f"Overall: {'ALL PASS ✓' if all_passed else 'SOME FAILURES — check retrieval quality'}")

    # Also show the formatted context for one query
    print("\n" + "=" * 60)
    print("FORMATTED CONTEXT EXAMPLE (for prompt injection):")
    print("=" * 60)
    demo_results = retrieve("What are the home loan fees?")
    print(format_context(demo_results))