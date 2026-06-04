# test_embed.py  (root dir, delete after)
import os
from dotenv import load_dotenv
from src.rag.embeddings import GeminiEmbeddings

load_dotenv()

emb = GeminiEmbeddings()
result = emb.embed_query("What are the home loan interest rates?")

print(f"Model:          {emb.model}")
print(f"Vector length:  {len(result)}")
print(f"First 5 values: {result[:5]}")
print("✓ Embedding call succeeded")