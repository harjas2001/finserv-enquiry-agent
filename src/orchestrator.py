"""
Phase 3 — Orchestrator Node
src/orchestrator.py
====================
The orchestrator is the first node every enquiry passes through. It does one
thing: classify the customer's query into one of four intent categories, write
that intent to state, and let LangGraph's conditional edge route to the correct
subagent.
 
Graph position:
    START → [orchestrator_node] → route_to_subagent() → subagent node
 
Intent categories and their destinations:
    "account"      → account_node     (tool call → mock account API)
    "product"      → product_node     (RAG pipeline from Phase 2)
    "complaint"    → complaint_node   (tool call + HITL escalation)
    "out_of_scope" → deflector_node   (polite refusal, minimal LLM)
 
This file exports two objects used by graph.py:
    orchestrator_node   — the LangGraph node function
    route_to_subagent   — the conditional edge routing function
 
Phase 1 connection:
    orchestrator_node is structurally identical to agent_node from Phase 1 —
    it reads state, calls Gemini, returns a partial dict. The difference is
    what it does with the response: instead of deciding whether to call a tool,
    it extracts an intent string that determines which node runs next.
 
    route_to_subagent is the direct equivalent of should_continue() from Phase 1.
    Where should_continue returned "tools" or "end", route_to_subagent returns
    one of four subagent keys. Same mechanism, four destinations instead of two.
"""

import os
import sys 
from dotenv import load_dotenv

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.state import EnquiryState

load_dotenv()


# CONFIG
GEMINI_MODEL = "gemini-2.5-flash"
VALID_INTENTS = {"account", "product", "complaint", "out_of_scope"}


#CLASSIFICATION PROMPT
SYSTEM_PROMPT = """\
You are an intent classifier for Clearwater Bank's automated customer service system.
 
Classify the customer query into exactly one of these four categories:
 
  account      — Questions about the customer's own accounts: balances, transactions,
                 statements, account history, recent payments, transfer status
 
  product      — Questions about Clearwater Bank's financial products and features:
                 loan interest rates, eligibility criteria, fee schedules, savings
                 account terms, how a product works in general
 
  complaint    — Complaints, disputes, service failures, requests to escalate to a
                 human agent, expressions of dissatisfaction with service or charges
 
  out_of_scope — Anything else: general financial knowledge questions, competitor
                 comparisons, requests for personalised investment advice, topics
                 unrelated to Clearwater Bank products or the customer's own account
 
Rules:
  - If the query fits multiple categories, choose the most specific one.
    Example: "Why has my savings bonus rate disappeared?" → complaint (not product),
    because the customer is expressing dissatisfaction, not asking how the rate works.
  - If the query is vague, ambiguous, or does not fit clearly, choose out_of_scope.
  - Reply with ONLY the category name — no punctuation, no explanation, no other text.\
"""


#ORCHESTRATOR NODE
def orchestrator_node(state: EnquiryState) -> dict:
    """
    LangGraph node — classifies intent and sets the routing field.
 
    Reads from state:   query
    Writes to state:    intent  (last-write-wins)
                        messages (add_messages reducer — appended, not replaced)
                        error    (only on failure)
 
    Returns a partial dict with exactly the fields this node changes.
    LangGraph merges it into the running EnquiryState:
      - intent:   replaces the initial "" value
      - messages: add_messages appends [SystemMessage, HumanMessage, AIMessage]
                  to the existing list (starts as [] → becomes 3 messages)
 
    On failure:
      Routes to "out_of_scope" so the deflector handles the customer gracefully,
      and writes the exception to state["error"] for the FastAPI layer to log.
    """

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set")
    
    query = state["query"]
    print(f"\n[ORCHESTRATOR] Classifying: '{query[:80]}'")

    try: 
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0, # set to 0 as we want the routing to be purley deterministic (classification)
            ),
        )   

        """response.text is for handling gemini 2.5 flash thinking model content blocks
        automatically, no extract_text() helper needed when using the google-genai
        SDK directly. Only needed with the LangChain wrapper"""

        raw = response.text.strip().lower()
        print(f"[ORCHESTRATOR] Raw intent: '{raw}'")

        # Validate and apply fallback: If Gemini ignores the output constraint in system prompt then just set basic message
        intent = raw if raw in VALID_INTENTS else "out_of_scope"
        if intent != raw:
            print(f"[ORCHESTRATOR] Unexpected format '{raw}' -> fallback: 'out_of_scope'") #logging purposes
        print(f"[ORCHESTRATOR] Intent: '{intent}' -> routing to {intent} subagent")

        # Build LangChain message objects for state["messages"].
        # We use the google-genai SDK for the API call but still construct
        # LangChain message types for the messages field, they're just data
        # containers. The add_messages reducer will append these three to
        # state["messages"], which starts as [] from make_initial_state().
        #
        # The AIMessage stores raw (what Gemini actually said) rather than intent
        # (the validated string). This way if there's a fallback, can see
        # exactly what Gemini returned vs. what the system decided.
        return {
            "intent": intent,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=query),
                AIMessage(content=raw),
            ],
        }
    
    except Exception as e:
        # Gemini call failed (network error, quota exceeded, etc.) This is all backend errors
        # Route to out_of_scope so the customer sees a graceful response.
        # Write the error to state for logging, FastAPI will return HTTP 500.
        print(f"[ORCHESTRATOR] ERROR — {type(e).__name__}: {e}")
        return {
            "intent": "out_of_scope",
            "error": f"Orchestrator classification failed: {e}",
            "messages": [],
        }
    

# ROUTING FUNCTION: The conditional edge
# How this connects to graph.py:
#
#   graph.add_conditional_edges(
#       "orchestrator",        ← source node
#       route_to_subagent,     ← this function
#       {                      ← routing map: return value → node name
#           "account":      "account_node",
#           "product":      "product_node",
#           "complaint":    "complaint_node",
#           "out_of_scope": "deflector_node",
#       }
#   )
#
# The routing map in graph.py decouples the intent string from the node name.

def route_to_subagent(state: EnquiryState) -> str:
    """
    Conditional edge function — maps state["intent"] to the next node key.
 
    Called by LangGraph after orchestrator_node completes. The return value
    is looked up in the routing map passed to add_conditional_edges() in graph.py.
 
    Returns: "account" | "product" | "complaint" | "out_of_scope"
    """
    intent = state.get("intent", "out_of_scope")
    if intent not in VALID_INTENTS:
        print(f"[ROUTER] Unexpected intent '{intent}' → deflecting")
        return "out_of_scope"
    print(f"[ROUTER] Routing '{intent}'")
    return intent


# TEST HARNESS
#   1. Raw Gemini response should always be a single bare category name — no
#      punctuation, no explanation. This confirms the output constraint works.
#   2. All clear-cut queries (tests 1–4) should classify correctly.
#   3. The edge case (test 5) is deliberately ambiguous — it may classify as
#      "product" or "complaint". Either is defensible; the expected answer is
#      "complaint" per the tie-breaking rule. If Gemini picks "product", that
#      is before-state evidence.
#   4. route_to_subagent returns the same string as intent on the happy path.

if __name__ == "__main__":
    from src.state import make_initial_state

    if not os.environ.get("GOOGLE_API_KEY"):
        print("ERROR: Google API key not set.")
        sys.exit(1)

    print("=" * 60)
    print("ORCHESTRATOR NODE — classification test")
    print("=" * 60)

    test_cases = [
        {
            "query":    "What is the current balance on my savings account?",
            "expected": "account",
            "note":     "clear account intent",
        },
        {
            "query":    "What are the current interest rates on your home loans?",
            "expected": "product",
            "note":     "clear product intent",
        },
        {
            "query":    "I want to make a formal complaint about unauthorised fees.",
            "expected": "complaint",
            "note":     "clear complaint intent",
        },
        {
            "query":    "Can you give me tips on how to invest my superannuation?",
            "expected": "out_of_scope",
            "note":     "personalised financial advice, out of scope",
        },
        {
            "query":    "Why has my savings bonus rate dropped? This is completely unacceptable.",
            "expected": "complaint",
            "note":     "EDGE CASE — mentions product (rate) but tone is complaint",
        },
        {
            "query":    "What is the weather like in Sydney today?",
            "expected": "out_of_scope",
            "note":     "completely unrelated, clean deflection",
        },
    ]

    passed = 0
    failed = 0

    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(test_cases)}] {tc['note']}")
        print(f"Query:    '{tc['query']}'")
        print(f"Expected: {tc['expected']}")
 
        state = make_initial_state(tc["query"], customer_id="TEST")
        result = orchestrator_node(state)
 
        intent    = result.get("intent", "")
        msg_count = len(result.get("messages", []))
        matched   = intent == tc["expected"]
 
        print(f"Got:      {intent}  {'✓ PASS' if matched else '✗ FAIL'}")
        print(f"Messages written to state: {msg_count}")
 
        # Simulate the routing function against the returned intent
        merged_state = {**state, **result}
        route = route_to_subagent(merged_state)
        print(f"Router returns: '{route}'")
 
        if matched:
            passed += 1
        else:
            failed += 1
 
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    if failed > 0:
        print("\nAny failures on edge cases → document for Slide 5.")
        print("Fix: add a few-shot example for the failing category to SYSTEM_PROMPT.")
        print("Capture before + after terminal output as rubric evidence.")
    print("=" * 60)