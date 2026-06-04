# scratch_evidence.py
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.subagents.product import answer_product_query

load_dotenv()

query = "What is the variable interest rate on a home loan?"

# ── BEFORE RAG: raw LLM, no context ──────────────────────────────────────────
print("── WITHOUT RAG (no context — training data only) ──")
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=query,
    config=types.GenerateContentConfig(temperature=0),
)
print(response.text.strip())

print()

# ── AFTER RAG: retrieved context + grounding instruction ──────────────────────
print("── WITH RAG + grounding instruction ──")
result = answer_product_query(query, use_grounding=True)
print(result["answer"])