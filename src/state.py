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




# ─────────────────────────────────────────────────────────────────────────────
# ENQUIRY STATE
# ─────────────────────────────────────────────────────────────────────────────



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

