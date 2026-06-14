"""
Phase 3 — Out-of-Scope Deflector Subagent
src/subagents/deflector.py
===========================
Handles queries the orchestrator has already classified as out_of_scope.
Produces a clear, professional deflection response that lists what the
system CAN help with and explicitly invites the user to try again.
 
Graph position:
    orchestrator_node → (intent="out_of_scope") → deflector_node → guardrail_node
 
Reads from state:   query (for context in the response), customer_id
Writes to state:    subagent_response, sources (always []),
                    escalated (always False), allow_retry (always True)
 
Design decision — no LLM call
───────────────────────────────
Every other subagent uses Gemini in some form. The deflector does not. Three
reasons:
 
1. We already know the intent is out_of_scope, the orchestrator classified it
   before routing here. There is nothing left to infer. A Gemini call would add
   latency and cost to produce a response that is structurally identical every
   time: "I can't help with that, here's what I can help with, please try again."
 
2. The architecture diagram explicitly notes "no LLM on refuse" for the deflector.
   A deterministic response is also a testable response, the test harness can
   assert exact string content without worrying about model output variance.
 
3. A clear, consistent, professional deflection is better than a cleverly-worded 
   one that could hallucinate in unexpected directions.
 
The allow_retry pattern
────────────────────────
setting allow_retry=True in the return dict doesn't create a loop inside the
graph, the graph still ends normally after the guardrail node. What it does:
 
  1. deflector_node sets allow_retry=True in state.
  2. guardrail_node passes through (no PII / hallucination risk in a template).
  3. FastAPI reads allow_retry from the final state and includes it in the
     HTTP response payload: {"answer": "...", "allow_retry": true}.
  4. The channel (frontend / contact centre platform) reads that flag and keeps
     the session open, it doesn't terminate the conversation.
  5. The user's next message triggers a fresh graph.invoke() with the same
     customer_id, a new enquiry, clean slate, same session context.
 
This is the same pattern contact centre platforms (Genesys, CCAI) use for
DTMF no-match and speech no-match re-prompts. The session lives at the channel
layer; the graph handles one turn at a time.
 
PRODUCTION Note (for Slide 9 / Slide 10):
In production want retry_count: int in state. After N consecutive
deflections in a session, route to a human agent rather than continuing to
deflect. Not implemented in this POC, flagging as an escalation safety net 
and a metric to monitor (deflection rate, consecutive-deflection rate).
"""

import os
import sys

from src.state import EnquiryState

#DEFLECTION RESPONSE
# Design principles:
#   1. Don't tell the user what they CAN'T do, tell them what they CAN do.
#   2. End with an open question, "What would you like to know?" keeps the
#      session alive conversationally ready to help.
#   3. No specifics about why the query was out-of-scope, dont want to feel
#      could feel dismissive or odd in edge cases.

DEFLECTION_RESPONSE = """\
I'm Clearwater Bank's virtual assistant FinServ and I'm not able to help with that particular request.
 
Here's what I can assist you with:
 
  • Account enquiries: check your balance, view recent transactions, or review your statement
  • Product information: home loan rates, personal loan terms, savings account features, and fee schedules
  • Complaints or disputes: log a concern, dispute a transaction, or request an escalation to a specialist
 
Is there something from the list above I can help you with today? Or would you like to speak to an agent?\
"""


#DEFLECTOR NODE
def deflector_node(state: EnquiryState) -> dict:
    query = state["query"],
    print(f"\n[DEFLECTOR] Query: '{query[:80]}'")
    print(f"[DEFLECTOR] Returning deflection response (no LLM call)")
    print(f"[DEFLECTOR] allow_retry = True")

    return {
        "subagent_response":    DEFLECTION_RESPONSE,
        "sources":              [],
        "escalated":            False,
        "allow_retry":          True,
    }


#TEST HARNESS
# What this checks:
#   1. Deflection response is returned for every out-of-scope query type
#   2. allow_retry=True in every case, session always stays open
#   3. escalated=False in every case, never routes to HITL from here
#   4. sources=[] in every case, no retrieval
#   5. Response content includes the three capability areas
#   6. The response ends with an invitation to try again

if __name__ == "__main__":
    from src.state import make_initial_state
 
    print("=" * 60)
    print("DEFLECTOR NODE — test harness (no API key needed)")
    print("=" * 60)
 
    # Representative sample of out-of-scope query types
    test_cases = [
        {
            "query":    "Can you give me stock tips for the ASX?",
            "note":     "financial advice — clearly out of scope",
        },
        {
            "query":    "What is the weather like in Sydney today?",
            "note":     "completely unrelated",
        },
        {
            "query":    "How does the Reserve Bank set interest rates?",
            "note":     "general financial knowledge — not Clearwater-specific",
        },
        {
            "query":    "What are CommBank's home loan rates?",
            "note":     "competitor question",
        },
        {
            "query":    "Can you write me a poem about banking?",
            "note":     "off-topic creative request",
        },
        {
            "query":    "xkcd",
            "note":     "single unintelligible token — extreme edge case",
        },
    ]
 
    passed  = 0
    failed  = 0
    checks  = [
        "allow_retry is True",
        "escalated is False",
        "sources is []",
        "response contains 'Account enquiries'",
        "response contains 'Product information'",
        "response contains 'Complaints or disputes'",
        "response ends with open question",
    ]
 
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(test_cases)}] {tc['note']}")
        print(f"Query: '{tc['query']}'")
 
        state  = make_initial_state(tc["query"])
        result = deflector_node(state)
 
        resp = result.get("subagent_response", "")
 
        # Run assertions
        assertions = [
            result.get("allow_retry") is True,
            result.get("escalated")   is False,
            result.get("sources")     == [],
            "Account enquiries"       in resp,
            "Product information"     in resp,
            "Complaints or disputes"  in resp,
            "?"                       in resp,          # ends with open question
        ]
 
        all_ok = all(assertions)
        for check, ok in zip(checks, assertions):
            status = "✓" if ok else "✗"
            print(f"  {status} {check}")
 
        if all_ok:
            passed += 1
        else:
            failed += 1
 
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)}")
    print()
    print("Deflection response shown below (same for all queries):")
    print("─" * 60)
    print(DEFLECTION_RESPONSE)
    print("─" * 60)
    if failed == 0:
        print("\n✓ Deflector ready. Next: src/subagents/product.py node wrapper,")
        print("  then full graph assembly in src/graph.py.")
    print("=" * 60)