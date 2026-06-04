"""
Phase 2 — Product Information Subagent
src/subagents/product.py
========================
Answers customer questions about Clearwater Bank financial products
using RAG over the product knowledge base.

Pipeline per query:
    retrieve → relevance gate → format context → grounded Gemini call → return

Phase 3 note:
    In Phase 3 this function gets wrapped in a LangGraph node:
        def product_node(state: EnquiryState) -> dict:
            result = answer_product_query(state["query"])
            return {"subagent_response": result["answer"], "sources": result["sources"]}
    The function signature stays unchanged — only the wrapper changes.

Run directly to test:
    python -m src.subagents.product
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.rag.retriever import format_context, retrieve_with_scores

load_dotenv()


# ── Configuration ─────────────────────────────────────────────────────────────

# Relevance threshold — ChromaDB returns cosine DISTANCE (lower = more similar).
# Calibrated from retriever test results:
#   In-scope queries:     0.52 – 0.68  (good match)
#   Out-of-scope query:   0.92         (no match)
# Anything above 0.80 means no useful context was found — deflect rather than guess.
RELEVANCE_THRESHOLD = 0.80

GEMINI_MODEL = "gemini-2.5-flash"


# ── Prompts ───────────────────────────────────────────────────────────────────
#
# GROUNDING INSTRUCTION — the core of the RAG pattern.
# This instruction forces the LLM to treat retrieved context as its only source
# of truth, preventing it from substituting training-data knowledge.
#
# Prompt engineering evidence (Slide 5):
# The test harness below runs the same query with and without this instruction.
# Without it, Gemini answers from training data — rates will be generic or wrong.
# With it, responses are grounded in Clearwater Bank's actual product guides.
# Screenshot both outputs for the before/after comparison required for the rubric.

SYSTEM_PROMPT = """\
You are a product information specialist for Clearwater Bank.

Your role is to answer customer questions about Clearwater Bank's financial
products accurately and helpfully.

RULES:
1. Answer ONLY using the information provided in the context section of the message.
   Do not use any information from your training data or general knowledge.
2. If the context does not contain enough information to fully answer the question,
   say: "I don't have that information in our current product guides. For assistance,
   please call us on 1300 555 100 or visit a branch."
3. Never provide financial advice. Present product information factually.
4. Quote rates, fees, and figures exactly as they appear in the context.
5. Keep responses concise and professional.\
"""

# Without grounding — used only to capture hallucination evidence for Slide 5.
SYSTEM_PROMPT_NO_GROUNDING = """\
You are a helpful banking assistant for Clearwater Bank.
Answer the customer's question about our products helpfully and in detail.\
"""


# ── Core subagent ─────────────────────────────────────────────────────────────

def answer_product_query(query: str, use_grounding: bool = True) -> dict:
    """
    Answer a product information query using RAG.

    Args:
        query:         The customer's question.
        use_grounding: Set False only when capturing hallucination evidence
                       for Slide 5. Always True in production.

    Returns a dict with:
        "answer"   — the response string to return to the customer
        "sources"  — list of source filenames used (empty if deflected)
        "grounded" — False if the query was deflected (out of scope)
        "score"    — top retrieval score (lower = better match; >0.80 = no match)
    """
    # Step 1 — Retrieve top-K chunks with relevance scores
    results = retrieve_with_scores(query)

    if not results:
        return {
            "answer": "I was unable to search the knowledge base. Please try again.",
            "sources": [],
            "grounded": False,
            "score": 1.0,
        }

    top_score = results[0]["score"]

    # Step 2 — Relevance gate: deflect if nothing in the KB is a good match
    # This is the hallucination guardrail — if we can't ground the answer, we
    # don't generate one. The LLM is never given a chance to make something up.
    if top_score > RELEVANCE_THRESHOLD:
        return {
            "answer": (
                "I can only assist with questions about Clearwater Bank's financial "
                "products — home loans, personal loans, savings accounts, and general "
                "borrowing topics. For other enquiries, please call 1300 555 100."
            ),
            "sources": [],
            "grounded": False,
            "score": top_score,
        }

    # Step 3 — Format retrieved chunks into a context block for prompt injection
    context = format_context(results)
    sources = sorted({r["source"] for r in results})   # deduplicated

    # Step 4 — Build the user turn: context + question
    # The context is placed in the user message (not the system prompt) so the
    # LLM treats it as ground truth for this specific query.
    user_prompt = f"""\
Context from Clearwater Bank product guides:

{context}

---

Customer question: {query}

Answer using only the context above.\
"""

    # Step 5 — Generate answer with Gemini
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    client = genai.Client(api_key=api_key)
    system = SYSTEM_PROMPT if use_grounding else SYSTEM_PROMPT_NO_GROUNDING

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0,       # deterministic — product info should never vary
        ),
    )

    return {
        "answer": response.text.strip(),
        "sources": sources,
        "grounded": True,
        "score": top_score,
    }


# ── Test harness ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PRODUCT INFORMATION SUBAGENT — TEST HARNESS")
    print("=" * 60)

    test_queries = [
        {
            "query": "What is the variable interest rate on a home loan?",
            "note":  "expects grounded answer with specific rate from home_loan_guide",
        },
        {
            "query": "Can I make extra repayments on my personal loan?",
            "note":  "expects grounded answer from personal_loan_facts",
        },
        {
            "query": "What are the conditions to earn the bonus interest on a savings account?",
            "note":  "expects grounded answer from savings_account_summary",
        },
        {
            "query": "What is the exchange rate for US dollars?",
            "note":  "expects out-of-scope deflection — nothing about FX in KB",
        },
    ]

    for tc in test_queries:
        query = tc["query"]
        print(f"\n{'─' * 60}")
        print(f"QUERY:  {query}")
        print(f"EXPECT: {tc['note']}")
        print(f"{'─' * 60}")

        result = answer_product_query(query)

        status = "GROUNDED" if result["grounded"] else "DEFLECTED (out of scope)"
        print(f"Score:   {result['score']:.4f}  [{status}]")
        if result["sources"]:
            print(f"Sources: {result['sources']}")
        print(f"\nAnswer:\n{result['answer']}")

    # ── Prompt engineering evidence capture ───────────────────────────────────
    # Capstone Slide 5 (4 pts): screenshot both outputs below.
    # They demonstrate what grounding does — the core argument for RAG over
    # a plain LLM call.

    print(f"\n{'=' * 60}")
    print("PROMPT ENGINEERING EVIDENCE — grounding instruction: ON vs OFF")
    print("Screenshot both outputs for Slide 5.")
    print("=" * 60)

    evidence_query = "What is the variable interest rate on a home loan?"
    print(f"\nQuery: '{evidence_query}'\n")

    print("── WITH grounding (production) ─────────────────────────────")
    print("LLM is constrained to context. Should cite the exact Clearwater rate.")
    grounded_result = answer_product_query(evidence_query, use_grounding=True)
    print(f"\n{grounded_result['answer']}")

    print(f"\n── WITHOUT grounding (hallucination risk) ──────────────────")
    print("LLM answers from training data. May cite generic or competitor rates.")
    ungrounded_result = answer_product_query(evidence_query, use_grounding=False)
    print(f"\n{ungrounded_result['answer']}")

    print(f"\n{'=' * 60}")
    print("Evidence capture complete. Save this terminal output.")
    print("=" * 60)