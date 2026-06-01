"""
Phase 2 — RAG Knowledge Base Ingestion Pipeline
src/rag/ingest.py

Run once (or whenever documents change) to populate ChromaDB.
Run from the project root:
    python -m src.rag.ingest

What this script does:
  1. Load  — reads all .pdf files from data/knowledge_base/ (one Document per page)
  2. Chunk — splits into 512-token chunks with 50-token overlap
  3. Embed — calls Gemini text-embedding-004 on each chunk
  4. Store — persists to ChromaDB at data/chroma_db/
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()  # loads .env if present; no-op if not
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# ── Configuration ────────────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")
CHROMA_PERSIST_DIR = "data/chroma_db"
COLLECTION_NAME    = "product_knowledge"

# Chunk size in tokens. 512 tokens ≈ 350–400 words — a substantial paragraph.
# The overlap (50 tokens) prevents a key concept from being cut mid-sentence
# and lost to retrieval because it sat at a chunk boundary.
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50


# ── Stage 1: Load ─────────────────────────────────────────────────────────────

def load_documents(directory: Path) -> list:
    """
    Load all .pdf files. PyPDFLoader produces one Document object PER PAGE:

      TextLoader  → 1 Document per file,  metadata: {source: "file.txt"}
      PyPDFLoader → 1 Document per page,  metadata: {source: "file.pdf", page: 0}

    Why per-page matters:
      - Chunks carry page numbers → the agent can cite "home_loan_guide.pdf, p.2"
      - Easier to debug retrieval — you know exactly where a chunk came from
      - Page-level metadata is useful in the eval harness

    We overwrite metadata["source"] to just the filename (PyPDFLoader sets it
    to the full path by default, which is noisy in logs and prompts).
    """
    documents = []
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        print(f"  WARNING: No .pdf files found in {directory}")
        return documents

    for file_path in pdf_files:
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()   # one Document per page

        for doc in pages:
            doc.metadata["source"] = file_path.name  # "home_loan_guide.pdf"
            # doc.metadata["page"] is set automatically by PyPDFLoader (0-indexed)

        documents.extend(pages)
        total_chars = sum(len(p.page_content) for p in pages)
        print(f"  ✓ {file_path.name:40s} {len(pages)} pages  {total_chars:>6,} chars")

    return documents


# ── Stage 2: Chunk ────────────────────────────────────────────────────────────

def chunk_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.

    Why RecursiveCharacterTextSplitter?
    ─────────────────────────────────
    It tries a sequence of separators in order, falling back to finer splits
    only when needed:
        1. "\n\n"  — paragraph boundaries (preferred)
        2. "\n"    — line breaks
        3. ". "    — sentence boundaries
        4. " "     — word boundaries (last resort)
        5. ""      — character split (absolute fallback)

    This is "semantic-aware" in that it respects the natural structure of the
    document. A naive character splitter would cut mid-sentence randomly.

    On chunk sizing:
    ────────────────
    We target 512 tokens with 50-token overlap. Gemini's tokenizer isn't
    available client-side. tiktoken (from_tiktoken_encoder) is the ideal
    proxy — use it on your own machine (it downloads the vocab on first use).

    In this environment we use a character approximation:
        512 tokens × ~4 chars/token ≈ 2048 chars
        50 tokens  × ~4 chars/token ≈  200 chars

    In practice the results are nearly identical — English prose averages
    3.8–4.2 chars per token for most tokenizers. Chunk boundaries land in
    the same places because RecursiveCharacterTextSplitter splits at
    paragraph/sentence boundaries regardless.

    To switch to exact token counting on your machine:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="cl100k_base",
            chunk_size=512,
            chunk_overlap=50,
            separators=[...],
        )
    """
    # 512 tokens * 4 chars/token = 2048 chars; 50 tokens * 4 = 200 chars overlap
    CHARS_PER_TOKEN = 4
    chunk_size_chars    = CHUNK_SIZE    * CHARS_PER_TOKEN   # 2048
    chunk_overlap_chars = CHUNK_OVERLAP * CHARS_PER_TOKEN   #  200

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=chunk_overlap_chars,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    return chunks


def print_chunk_stats(chunks: list) -> None:
    """Print a breakdown of chunks per source document."""
    from collections import Counter
    source_counts = Counter(c.metadata.get("source", "unknown") for c in chunks)
    print(f"\n  Chunk distribution:")
    for source, count in sorted(source_counts.items()):
        print(f"    {source:40s} → {count:3d} chunks")
    print(f"    {'TOTAL':40s} → {len(chunks):3d} chunks")


# ── Stage 3: Embed + Store ────────────────────────────────────────────────────

def build_vectorstore(chunks: list) -> Chroma:
    """
    Generate embeddings for every chunk and persist to ChromaDB.

    What Gemini text-embedding-004 does:
    ─────────────────────────────────────
    Takes a text string → returns a list of 768 floats.
    That list is a point in 768-dimensional space. Semantically similar
    texts end up as nearby points. When we search later, we embed the
    query the same way and find the nearest chunk vectors.

    What ChromaDB stores (per chunk):
    ──────────────────────────────────
    - id:        auto-generated UUID
    - document:  the chunk text (used at retrieval time to inject into prompt)
    - embedding: the 768-dim vector (used for similarity search)
    - metadata:  {"source": "filename.txt"}  (passed through to results)

    ChromaDB persists everything to a local SQLite + on-disk index at
    CHROMA_PERSIST_DIR. The next time we load Chroma(..., persist_directory=...),
    it reads from disk — no need to re-embed.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\n  ERROR: GOOGLE_API_KEY not set.")
        print("  Export it: export GOOGLE_API_KEY='your-key-here'")
        sys.exit(1)

    print(f"  Embedding model : text-embedding-004")
    print(f"  Vector dimensions: 768")
    print(f"  Chunks to embed  : {len(chunks)}")
    print(f"  Persisting to    : {CHROMA_PERSIST_DIR}/")
    print()

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key,
    )

    # Chroma.from_documents:
    #   For each chunk, calls embeddings.embed_documents([chunk.page_content])
    #   then inserts (text, vector, metadata) into the collection.
    #   If the directory already exists, this will ADD to the collection.
    #   To re-ingest from scratch, delete data/chroma_db/ first.
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION_NAME,
    )

    return vectorstore


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RAG INGESTION — finserv-enquiry-agent")
    print("=" * 60)

    # ── Stage 1: Load ──────────────────────────────────────────────
    print(f"\n[1/3] Loading documents from {KNOWLEDGE_BASE_DIR}/")
    if not KNOWLEDGE_BASE_DIR.exists():
        print(f"  ERROR: {KNOWLEDGE_BASE_DIR} not found.")
        print("  Make sure you are running from the project root.")
        sys.exit(1)

    documents = load_documents(KNOWLEDGE_BASE_DIR)
    total_chars = sum(len(d.page_content) for d in documents)
    print(f"\n  Total: {len(documents)} documents | {total_chars:,} characters")

    # ── Stage 2: Chunk ─────────────────────────────────────────────
    print(f"\n[2/3] Chunking  (size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP} tokens)")
    chunks = chunk_documents(documents)
    print_chunk_stats(chunks)

    # Show a sample chunk so we can verify quality
    sample = chunks[2]  # pick chunk 2 to skip the header chunk
    print(f"\n  Sample chunk (index 2) from '{sample.metadata['source']}':")
    print("  " + "─" * 50)
    print("  " + sample.page_content[:400].replace("\n", "\n  "))
    print("  " + "─" * 50)

    # ── Stage 3: Embed + Store ─────────────────────────────────────
    print(f"\n[3/3] Embedding + storing to ChromaDB")
    vectorstore = build_vectorstore(chunks)

    total_in_db = vectorstore._collection.count()
    print(f"\n✓ Ingestion complete.")
    print(f"  {total_in_db} chunks in ChromaDB collection '{COLLECTION_NAME}'")
    print(f"  Location: {CHROMA_PERSIST_DIR}/")
    print(f"\n  Next step: run  python -m src.rag.retriever  to test queries.")


if __name__ == "__main__":
    main()