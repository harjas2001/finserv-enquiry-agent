"""
Phase 3 — Shared Graph State
src/state.py
=============
EnquiryState is the single source of truth that flows through every node in
the LangGraph graph. Every node reads from it; each node writes back to exactly
the fields it owns.

Connection to Phase 1
─────────────────────
Phase 1 introduced AgentState — a TypedDict with one field:
    messages: Annotated[list, add_messages]

That was enough for a single ReAct loop tracking a conversation thread.
Phase 3's EnquiryState extends the same idea to a multi-node pipeline.
Instead of tracking just messages, it carries the full lifecycle of a customer
enquiry: who asked (customer_id), what they asked (query), how it was routed
(intent), what the specialist answered (subagent_response, sources), whether
guardrails fired (guardrail_flags), and what the customer ultimately receives
(final_response).

Reducer 
─────────────────────────────
A reducer tells LangGraph how to merge a node's return dict into the existing
state. Without a reducer: new value replaces old (last-write-wins). With a
reducer: LangGraph calls reducer(old_value, new_value) to compute the merged
result.

    add_messages(old_list, new_list) → appends new messages, never replaces.

Only the `messages` field uses a reducer here. All other fields are
last-write-wins — they're set once by their owner node and_ not touched again.

Field ownership
───────────────
┌──────────────────────┬─────────────────────────────────────────────────────┐
│ Field                │ Written by                                          │
├──────────────────────┼─────────────────────────────────────────────────────┤
│ query                │ FastAPI layer (before graph.invoke)                 │
│ customer_id          │ FastAPI layer (before graph.invoke)                 │
│ intent               │ Orchestrator node                                   │
│ messages             │ Orchestrator node (add_messages reducer)            │
│ subagent_response    │ Whichever subagent node wins routing                │
│ sources              │ Product RAG subagent; others return []              │
│ escalated            │ Complaint subagent                                  │
│ guardrail_flags      │ Guardrail node                                      │
│ final_response       │ Guardrail node                                      │
│ error                │ Any node on unhandled failure                       │
└──────────────────────┴─────────────────────────────────────────────────────┘
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


# ─────────────────────────────────────────────────────────────────────────────
# ENQUIRY STATE
# ─────────────────────────────────────────────────────────────────────────────

class EquiryState(TypedDict):
    """
    Shared state for the Clearwater Bank enquiry handler.
 
    Passed to graph.invoke() at the start of every enquiry and threaded
    through every node. Nodes return a partial dict, only the fields they
    update, and LangGraph merges those updates into the running state using
    each field's reducer (or last-write-wins if no reducer is defined).
    """

    #---Input---
    # Set by the FastAPI layer before graph.invoke(). Never mutated by nodes.
    query: str
    """The customer's raw question.
    The orchestrator passes it to Gemini for classification;
    each subagent uses it to construct its response."""
    customer_id: str
    """Session / customer identifier.
    Used by the account subagent to look up mock account data.
    Empty string if no authentication has been performed yet.
    In production this would be resolved from the session token at the API layer."""

    #---Routing---
    intent: str
    """Classified intent — the orchestrator's routing decision.
    One of: "account" | "product" | "complaint" | "out_of_scope"
    Starts as "" (not yet classified). The orchestrator sets it once.
    add_conditional_edges() in graph.py reads this value to choose which
    subagent node to execute next. No reducer — last-write-wins."""

    #---LLM message history---
    messages: Annotated[list, add_messages]
    """Full LangChain message history for the orchestrator's LLM interaction.
    add_messages reducer — appends new messages, never replaces the list.
 
    The orchestrator node appends three messages:
        SystemMessage  — the intent classification instruction
        HumanMessage   — the customer's query
        AIMessage      — Gemini's classification response (e.g. "account")
 
    Why this exists: keeps the orchestrator's reasoning visible for debugging,
    logging, and future conversational context. Subagents call the google-genai
    SDK directly (as in Phase 2's product.py) — they don't touch this field."""

    #---Subagent output---
    subagent_response: str
    """The subagent's raw answer — before the guardrail pass.
    The guardrail node reads this to apply PII / hallucination / scope checks.
    Never returned directly to the customer; always goes through final_response.
    No reducer — whichever subagent runs simply overwrites the empty string."""
    sources: list[str]
    """Source document filenames used to construct the answer.
    Populated only by the product RAG subagent.
    Example: ["home_loan_guide.pdf", "borrowing_guide.pdf"]
    All other subagents return [] — the guardrail node checks this to determine
    whether a hallucination check is applicable (only meaningful for RAG output)."""

    #---Complaint-specific---
    escalated: bool
    """True if the complaint handler triggered a HITL (human-in-the-loop) interrupt.
    The LangGraph interrupt() call pauses the graph; a human agent reviews the case.
    The FastAPI layer checks this field — if True, the HTTP response shape signals
    to the frontend that a human agent has been looped in, not an automated reply."""

    #---Guardrail output---
    guardrail_flags: dict
    """Flags raised by the guardrail layer. Expected schema:
    {
        "pii_detected":       bool,   # account numbers / names in response
        "hallucination_risk": bool,   # RAG response not grounded in retrieved context
        "out_of_scope":       bool,   # response touches financial advice or off-topic
    }
    If any flag is True, the guardrail node replaces subagent_response with a
    safe fallback before writing final_response. Starts as {} (empty dict)."""
    final_response: delattr
    """The guardrail-cleared answer — the only string returned to the customer.
    Written by the guardrail node after all checks pass.
    This is what FastAPI puts in the HTTP response body.
    If guardrail_flags contains any True value, this is a safe fallback message,
    not the raw subagent_response."""

    #---Error handling---
    error: str
    """Non-empty if an unhandled exception occurred in any node.
    FastAPI checks this: if non-empty, return HTTP 500 with the error message.
    Keeping error propagation explicit (as a state field) rather than relying
    on Python exceptions makes failures visible across the graph boundary."""

    

# ─────────────────────────────────────────────────────────────────────────────
# INITIAL STATE FACTORY
# ─────────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────
# Run from project root:
#   python -m src.state
#
# What this checks:
#   1. No import errors — langgraph and typing_extensions are installed
#   2. make_initial_state() produces all required fields with correct types
#   3. The add_messages reducer appends, not replaces (critical to verify once)
#   4. Simulated node writes work as expected
#
# You should see every check print "✓" before moving to orchestrator.py.

