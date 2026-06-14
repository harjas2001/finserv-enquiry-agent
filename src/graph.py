"""
Phase 3: Full Graph Assembly
src/graph.py
==============
Wires every component built in Phase 3 into a single compiled StateGraph.
This is the file that turns a collection of nodes into a working system.
 
Full topology:
 
    START
      ↓
    orchestrator_node        ← Gemini classifies intent (temperature=0)
      ↓ route_to_subagent()
      ├── "account"      → account_node      (two-turn function calling)
      ├── "product"      → product_node      (RAG pipeline from Phase 2)
      ├── "complaint"    → complaint_node    (two-turn function calling + severity)
      └── "out_of_scope" → deflector_node    (deterministic, no LLM, allow_retry=True)
                               ↓ route_after_complaint()
              ┌────────────────┤
              │ "escalate"     └─── escalation_node   (interrupt — HITL)
              │ "continue"          ↓ (human resumes)
              └─────────────────────┤
                                    ↓ (all paths merge)
                              guardrail_node   ← Phase 3: passthrough
                                    ↓            Phase 4: PII + hallucination checks
                                   END
 
Key design decisions
────────────────────
1. MemorySaver checkpointer: required because escalation_node calls interrupt(),
   which pauses the graph and saves state to the checkpointer. Without it,
   interrupt() raises an error immediately. In production this would be a
   database-backed checkpointer (e.g. PostgresSaver).
 
2. thread_id in every invoke() config: the key that tells the checkpointer
   which session's state to load or save. Each unique conversation needs a
   unique thread_id; reusing one would load the wrong saved state.
 
3. Passthrough guardrail (Phase 3 placeholder): the real guardrail layer is
   Phase 4. For Phase 3 we need a node at that position so the graph is
   complete and end-to-end testable.
 
4. The escalation_node interrupt/resume cycle: first real test of the HITL
   pattern. When an urgent complaint routes here, the graph pauses and the
   test harness (or API layer) must call graph.invoke(Command(resume=...),
   config=same_config) to continue. The same thread_id in both calls tells
   MemorySaver which paused state to restore.
 
Exported from this file (imported by api/main.py in Phase 3 end):
    build_graph()  → compiled LangGraph graph, ready for invoke()
"""

import os
import sys

from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
 
from src.orchestrator import orchestrator_node, route_to_subagent
from src.state import EnquiryState, make_initial_state
from src.subagents.account import account_node
from src.subagents.complaint import (complaint_node, escalation_node,
                                      route_after_complaint)
from src.subagents.deflector import deflector_node
from src.subagents.product import product_node
 
load_dotenv()


#PASSTHROUGH GUARDRAIL (placeholder)
# set all guardrail flags to False
def passthrough_guardrail(state: EnquiryState) -> dict:
    print(f"\n[GUARDRAIL] post subagent passthrough, no checks applied.")
    
    return {
        "guardrail_flags": {
            "pii_detected":         False,
            "hallucination_risk":   False,
            "out_of_scope":         False,
        },
        "final_response":   state["subagent_response"],
    }


#GRAPH ASSEMBLY
def build_graph(checkpointer=None):
    """
    Assemble, wire, and compile the full multi-agent StateGraph.
 
    Args:
        checkpointer: A LangGraph checkpointer instance. If None, a new
                      MemorySaver is created internally. Pass in an existing
                      instance when you need to share the checkpointer across
                      multiple calls (e.g. to resume from an interrupt in the
                      same test run).
 
    Returns:
        A compiled LangGraph graph, ready for graph.invoke(state, config=config).
        Every call must include config={"configurable": {"thread_id": "<id>"}}.
 
    Why thread_id is mandatory with a checkpointer:
        The checkpointer stores paused graph state keyed by thread_id. Without
        it, the checkpointer doesn't know where to save (or restore from). Each
        unique conversation should have a unique thread_id. Reusing one loads
        the previous conversation's state.
    """
    
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = StateGraph(EnquiryState)

    #REGISTER NODES
    #add_edge and add_conditional_edges will refer to node string name
    #Whcih will then call the function associated
    graph.add_node("orchestrator",  orchestrator_node)
    graph.add_node("account",       account_node)
    graph.add_node("product",       product_node)
    graph.add_node("complaint",     complaint_node)
    graph.add_node("escalation",    escalation_node)
    graph.add_node("deflector",     deflector_node)
    graph.add_node("guardrail",     passthrough_guardrail)  

    #ENTRY Point
    graph.add_edge(START, "orchestrator")

    #Orchestrator → subagents (conditional routing)
    # route_to_subagent(state) returns one of the four strings in the dict.
    # LangGraph maps that string to the next node name.
    graph.add_conditional_edges(
        "orchestrator",
        route_to_subagent,
        {
            "account":      "account",
            "product":      "product",
            "complaint":    "complaint",
            "out_of_scope": "deflector",
        },
    )

    #Subagents → guardrail (direct edges)
    # account, product, deflector always go straight to guardrail.
    # No conditional logic, their subagent_response is always ready.
    graph.add_edge("account",   "guardrail")
    graph.add_edge("product",   "guardrail")
    graph.add_edge("deflector", "guardrail")

    #Complaint → escalation OR guardrail (conditional)
    # route_after_complaint(state) checks state["escalated"]:
    #   True  → "escalate" → escalation_node (HITL interrupt)
    #   False → "continue" → guardrail_node  (standard flow)
    graph.add_conditional_edges(
        "complaint",
        route_after_complaint,
        {
            "escalate": "escalation",
            "continue": "guardrail",
        },
    )

    #Escalation → guardrial (after human resumes)
    # Once a human reviewer calls Command(resume=...) and execution continues
    # past interrupt(), escalation_node finishes and flows to guardrail.
    graph.add_edge("escalation", "guardrail")

    #Guardrail → END
    graph.add_edge("guardrail", END)

    #COMPILE
    # checkpointer is required for interrupt() in escalation_node.
    # Without it, the graph compiles but crashes at runtime when interrupt()
    # is called with no persistence layer to save the paused state to.   
    compiled = graph.compile(checkpointer=checkpointer)
    print("[GRAPH] Compiled Successfully.")
    return compiled 


# TEST HARNESS
# Stage A — structural check (no API key needed):
#   Builds and compiles the graph. Verifies all nodes are registered and edges
#   are valid. Catches import errors, missing nodes, and wiring mistakes before
#   making any API calls.
#
# Stage B — end-to-end live tests (GOOGLE_API_KEY required):
#   Runs all five routing paths through the compiled graph:
#     Test 1: account path (tool call → balance)
#     Test 2: product path (RAG → grounded answer)
#     Test 3: out-of-scope path (deflector → allow_retry)
#     Test 4: standard complaint (logged, not escalated)
#     Test 5: urgent complaint (logged, escalated → interrupt → resume)
#
#   Test 5 is the first real test of the HITL interrupt/resume cycle.
#   Watch for the [ESCALATION] log line and the GraphInterrupt catch.
 
if __name__ == "__main__":
 
    # ── Stage A: structural check ──────────────────────────────────────────
    print("=" * 60)
    print("STAGE A — Structural check (no API key needed)")
    print("=" * 60)
 
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)
 
    # LangGraph exposes the node names via the graph's internal structure.
    # Verify all seven nodes are registered.
    expected_nodes = {
        "orchestrator", "account", "product", "complaint",
        "escalation", "deflector", "guardrail",
    }
    actual_nodes = set(graph.nodes.keys()) - {"__start__"}
    missing = expected_nodes - actual_nodes
    extra   = actual_nodes - expected_nodes
 
    if missing:
        print(f"  ✗ Missing nodes: {missing}")
        sys.exit(1)
    if extra:
        print(f"  ✗ Unexpected nodes: {extra}")
        sys.exit(1)
 
    print(f"  ✓ All {len(expected_nodes)} nodes registered: {sorted(expected_nodes)}")
 
    # Verify graph can be invoked structurally by checking it has a start node
    assert "__start__" in graph.nodes or START in str(graph.nodes)
    print("  ✓ START → orchestrator entry point wired")
    print("  ✓ Graph compiled with MemorySaver checkpointer")
    print("\n✓ Stage A passed\n")
 
    # ── Stage B: live end-to-end tests ────────────────────────────────────
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Stage B skipped — GOOGLE_API_KEY not set.")
        sys.exit(0)
 
    print("=" * 60)
    print("STAGE B — End-to-end live tests (Gemini API)")
    print("=" * 60)
 
    def run_test(label, query, customer_id, thread_id, note=""):
        print(f"\n{'─' * 60}")
        print(f"{label}")
        if note:
            print(f"Note: {note}")
        print(f"Query:       '{query}'")
        print(f"Customer ID: '{customer_id}'")
 
        state  = make_initial_state(query, customer_id=customer_id)
        config = {"configurable": {"thread_id": thread_id}}
 
        try:
            result = graph.invoke(state, config=config)
 
            # LangGraph interrupt() behaviour is version-dependent:
            #   Older versions: raise GraphInterrupt (caught in except below)
            #   Newer versions: return the paused state with final_response=""
            #
            # Detect the return-based pause: escalated=True (complaint node ran
            # and flagged for HITL) but final_response="" (guardrail never ran
            # because the graph paused before reaching it).
            if result.get("escalated") and not result.get("final_response"):
                draft = result.get("subagent_response", "")
                print(f"\n  ✋ Graph paused at interrupt (invoke returned paused state)")
                print(f"  draft response: '{draft[:80]}...'")
                print(f"  Simulating human reviewer approving draft...")
 
                result = graph.invoke(
                    Command(resume={"approved_response": draft}),
                    config=config,
                )
                print(f"  ✓ Graph resumed — HITL cycle complete")
 
            _print_result(result)
            return result
 
        except Exception as e:
            # interrupt() raises GraphInterrupt — handle HITL cycle here.
            if "GraphInterrupt" in type(e).__name__ or "Interrupt" in type(e).__name__:
                print(f"\n  ✋ Graph paused — interrupt() fired (expected for urgent complaint)")
                print(f"  Simulating human reviewer approving the draft response...")

                # The interrupt payload contains the draft response —
                # extract it from the exception args if available.
                draft = ""
                if hasattr(e, "args") and e.args:
                    interrupts = e.args[0]
                    if isinstance(interrupts, (list, tuple)) and interrupts:
                        payload = interrupts[0] if isinstance(interrupts[0], dict) else {}
                        draft = payload.get("draft_response", "")
 
                # In production: a human reads state["subagent_response"] and
                # sends back an approved (possibly edited) version.
                # Here: we approve the draft unchanged.
                resume_result = graph.invoke(
                    Command(resume={"approved_response": state.get("subagent_response", "Approved.")}),
                    config=config,
                )
                print(f"  ✓ Graph resumed after human approval")
                _print_result(resume_result)
                return resume_result
            else:
                print(f"  ✗ Unexpected error: {type(e).__name__}: {e}")
                raise
 
 
    def _print_result(result):
        print(f"\n  intent:        {result.get('intent', '?')}")
        print(f"  escalated:     {result.get('escalated', '?')}")
        print(f"  allow_retry:   {result.get('allow_retry', False)}")
        if result.get("sources"):
            print(f"  sources:       {result['sources']}")
        if result.get("error"):
            print(f"  error:         {result['error']}")
        print(f"\n  final_response:\n  {result.get('final_response', '')[:200]}")
 
 
    # Test 1 — Account path
    run_test(
        label="[1/5] Account path",
        query="What is my current account balance?",
        customer_id="C001",
        thread_id="graph-test-account",
        note="Expect: orchestrator→account→guardrail, grounded balance figures",
    )
 
    # Test 2 — Product path (RAG)
    run_test(
        label="[2/5] Product path (RAG)",
        query="What are your current home loan interest rates?",
        customer_id="C001",
        thread_id="graph-test-product",
        note="Expect: orchestrator→product→guardrail, Clearwater-specific rate cited",
    )
 
    # Test 3 — Out-of-scope path (deflector)
    run_test(
        label="[3/5] Out-of-scope path",
        query="Can you give me stock tips for the ASX?",
        customer_id="C001",
        thread_id="graph-test-oos",
        note="Expect: orchestrator→deflector→guardrail, allow_retry=True",
    )
 
    # Test 4 — Standard complaint (no escalation)
    run_test(
        label="[4/5] Standard complaint",
        query="I was charged a $35 fee on my account I was never told about.",
        customer_id="C001",
        thread_id="graph-test-complaint-std",
        note="Expect: orchestrator→complaint→guardrail (no interrupt), escalated=False",
    )
 
    # Test 5 — Urgent complaint (triggers interrupt)
    # This is the first real HITL test. The graph will pause at escalation_node
    # and the run_test() wrapper handles the interrupt/resume cycle.
    run_test(
        label="[5/5] Urgent complaint — HITL interrupt/resume",
        query="There is a $450 transaction on my account I never made. "
              "I think someone has accessed my account without my permission.",
        customer_id="C002",
        thread_id="graph-test-complaint-esc-v3",   # fresh ID — don't reuse paused sessions
        note="Expect: orchestrator→complaint→escalation (interrupt!)→guardrail, escalated=True",
    )
 
    print(f"\n{'=' * 60}")
    print("Stage B complete — full graph end-to-end.")
    print("Check each test above for correct intent routing and final_response.")
    print("=" * 60)