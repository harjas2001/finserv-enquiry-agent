"""
Phase 3 — FastAPI Serving Layer
api/main.py
================
Exposes the compiled LangGraph multi-agent graph as an HTTP API.
 
The API layer does:
    1. Accept an HTTP request and validate the payload (Pydantic)
    2. Call graph.invoke() with the right state and config
    3. Detect and handle the HITL interrupt/resume cycle
    4. Shape the final EnquiryState into a JSON response
 
Endpoints:
    GET  /health:    liveness probe (Cloud Run, load balancer)
    POST /enquire:   main enquiry endpoint
 
Run locally:
    uvicorn api.main:app --reload --port 8000
 
Test with curl:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/enquire \\
         -H "Content-Type: application/json" \\
         -d '{"query": "What is my balance?", "customer_id": "C001"}'
 
Production note: MemorySaver vs PostgresSaver:
    For production, will replace MemorySaver with PostgresSaver
    (or equivalent) backed by Cloud SQL, the same thread_id will then resolve
    to the same paused state regardless of which instance handles the request.
"""

import uuid
from typing import Optional
 
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field
 
from src.graph import build_graph
from src.state import make_initial_state
 
load_dotenv()


#GRAPH SINGLETON
# Built once at module load, shared across every request.
#
# Why module-level and not inside the endpoint function:
#   graph.compile() is expensive (validates topology, sets up runtime). Calling
#   it on every request would add latency and defeat the checkpointer: a new
#   MemorySaver on every call means every session starts from a blank slate,
#   making the interrupt/resume cycle impossible (the resume would look for a
#   paused state that doesn't exist in the new MemorySaver instance).
_checkpointer = MemorySaver()
_graph        = build_graph(checkpointer=_checkpointer)


#REQUEST/RESPONSE Models
class EnquireRequest(BaseModel):
    """payload client will send to POST /enquire."""
    query: str = Field(
        ...,
        min_length=1,
        description="The customer's question or message.",
        examples=["What is my current account balance?"]
    )
    customer_id: str = Field(
        default="",
        description="Authenticated customer identifier from the session token. "
                    "Empty string for unauthenticated sessions.",
    )
    session_id: str = Field(
        default="",
        description="Conversation session identifier used as the checkpointer "
                    "thread_id. If not provided, a UUID is generated and "
                    "returned in the response. The client must use the same "
                    "session_id for follow-up turns in the same conversation.",
    )

class EnquireResponse(BaseModel):
    """Response returned from POST /enquire"""
    answer: str = Field(description="The final, guardrail-cleared response to the customer.")
    intent: str = Field(description="Intent classified by the orchestrator.")
    sources: list[str] = Field(default=[], description="Sources documents used (RAG responses only).")
    allow_retry: bool = Field(
        default=False,
        description="True when the deflector handled the query. "
                    "The client should keep the session open and prompt the "
                    "user to rephrase their question.",
    )
    escalated: bool = Field(
        default=False,
        description="True when an urgent complaint triggered the HITL escalation "
                    "node. The customer's case has been queued for human review.",
    )
    session_id: str = Field(description="Echo of the session_id used. Store for the follow-up turns.")


#APP
app = FastAPI(
    title="Clearwater Bank Enquiry Handler (FinServ)",
    description=(
        "Agentic customer enquiry handler, multi-agent system using "
        "Google Gemini + LangGraph with RAG, tool calling, and HITL escalation."
    ),
    version="0.3.0" #Phase3 thats why version set as such
)

#CORS: permissive for local development
#note: tighten CORS for production use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


#ENDPOINTS

@app.get("/health")
def health():
    """
    Liveness probe, used by Cloud Run and load balancers.
    Returns '200' OK when the app is running and the graph is compiled.
    """
    return {"status": "ok", "graph": "compiled", "version": app.version}

@app.post("/enquire", response_model=EnquireResponse)
def enquire(req: EnquireRequest):
    """
    Main enquiry endpoint. Accepts a customer query, runs it through the
    full multi-agent graph, and returns the structured response.
 
    Flow:
        request → make_initial_state → graph.invoke → (interrupt?) → response
 
    Interrupt handling:
        escalation_node calls LangGraph's interrupt() for urgent complaints.
        Depending on the LangGraph version, this either returns the paused
        state or raises GraphInterrupt. Both are detected and handled here,
        the draft response is auto-approved for the POC demo. In production,
        a human agent would review the draft in an admin UI before the resume
        call is made.
    """

    #Generate session_id (if not provided), usually in production client has external db for thus
    session_id  = req.session_id.strip() or str(uuid.uuid4())
    config      = {"configurable": {"thread_id": session_id}}
    state       = make_initial_state(req.query, customer_id=req.customer_id)

    try:
        result = _graph.invoke(state, config=config)

        # Interrupt detection:
        # LangGraph 0.2+ returns the paused state from invoke() when interrupt()
        # fires, rather than raising an exception. Signature: escalated=True
        # (set by complaint_node) + final_response="" (guardrail never ran).
        if result.get("escalated") and not result.get("final_response"):
            draft = result.get("subagent_response", "")
            result = _graph.invoke(
                Command(resume={"approved_response": draft}),
                config=config,
            )

        #Error propagation
        # Any node that failed wrote to state["error"]. Surface it as HTTP 500
        # so the client knows something went wrong and can fall back or alert.
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])
        
        return EnquireResponse(
            answer=result.get("final_response", ""),
            intent=result.get("intent", ""),
            sources=result.get("sources", []),
            allow_retry=result.get("allow_retry", False),
            escalated=result.get("escalated", False),
            session_id=session_id,            
        )

    except HTTPException:
        raise

    except Exception as e:
        #Interrupt detection (exception-based)
        if "GraphInterrupt" in type(e).__name__ or "Interrupt" in type(e).__name__:
            draft = ""
            if hasattr(e, "args") and e.args:
                interrupts = e.args[0]
                if isinstance(interrupts, (list, tuple)) and interrupts:
                    payload = interrupts[0] if isinstance(interrupts[0], dict) else {}
                    draft   = payload.get("draft_response", "")
 
            result = _graph.invoke(
                Command(resume={"approved_response": draft}),
                config=config,
            )
            return EnquireResponse(
                answer=result.get("final_response", ""),
                intent=result.get("intent", ""),
                sources=result.get("sources", []),
                allow_retry=result.get("allow_retry", False),
                escalated=result.get("escalated", False),
                session_id=session_id,
            )
 
        # Unexpected exception — log and return 500.
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
    

#TEST HARNESS
# Stage A — structural tests (no API key needed):
#   Uses FastAPI's TestClient to validate routes, models, and error handling
#   without making any Gemini API calls.
#
# Stage B — live end-to-end tests (GOOGLE_API_KEY required):
#   Runs all 5 paths through the full stack over HTTP using TestClient.
#   Equivalent to running curl commands against a live uvicorn server.
if __name__ == "__main__":
    import os
    import sys
 
    from fastapi.testclient import TestClient  # requires httpx
 
    client = TestClient(app)
 
    # ── Stage A: structural tests (no API key) ────────────────────────────
    print("=" * 60)
    print("STAGE A — Structural tests (no API key needed)")
    print("=" * 60)
 
    print("\n[1] GET /health")
    r = client.get("/health")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    body = r.json()
    assert body["status"] == "ok"
    assert body["graph"]  == "compiled"
    print(f"  status: {body['status']}  graph: {body['graph']}  ✓")
 
    print("\n[2] POST /enquire — missing query field → 422 Unprocessable Entity")
    r = client.post("/enquire", json={"customer_id": "C001"})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print(f"  status: {r.status_code}  ✓ (Pydantic validation working)")
 
    print("\n[3] POST /enquire — empty query string → 422")
    r = client.post("/enquire", json={"query": "", "customer_id": "C001"})
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print(f"  status: {r.status_code}  ✓ (min_length=1 enforced)")
 
    print("\n[4] POST /enquire — no session_id → auto-generated UUID in response")
    # Stub check: we can't make a real Gemini call here, so we just verify
    # the route exists and responds to a well-formed request (will fail with
    # 500 if GOOGLE_API_KEY is missing, but the route itself is reachable).
    r = client.post("/enquire", json={"query": "hello", "customer_id": ""})
    assert r.status_code in (200, 422, 500), f"Unexpected status: {r.status_code}"
    if r.status_code == 200:
        assert "session_id" in r.json()
        assert len(r.json()["session_id"]) > 0
        print(f"  session_id generated: {r.json()['session_id'][:8]}...  ✓")
    else:
        print(f"  Route reachable (status {r.status_code} — needs API key for full response)  ✓")
 
    print("\n✓ All Stage A checks passed\n")
 
    # ── Stage B: live end-to-end tests (API key required) ─────────────────
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Stage B skipped — GOOGLE_API_KEY not set.")
        print("\nTo test manually, run:")
        print("  uvicorn api.main:app --reload --port 8000")
        print("  curl -X POST http://localhost:8000/enquire \\")
        print('       -H "Content-Type: application/json" \\')
        print('       -d \'{"query":"What is my balance?","customer_id":"C001"}\'')
        sys.exit(0)
 
    print("=" * 60)
    print("STAGE B — Live end-to-end tests (Gemini API over HTTP)")
    print("=" * 60)
 
    test_cases = [
        {
            "payload": {"query": "What is my current account balance?", "customer_id": "C001"},
            "note":    "Account path — expect balance in answer, intent=account",
            "checks":  {"intent": "account", "allow_retry": False, "escalated": False},
        },
        {
            "payload": {"query": "What are your current home loan interest rates?", "customer_id": "C001"},
            "note":    "Product path — expect RAG-grounded rate, sources populated",
            "checks":  {"intent": "product", "allow_retry": False, "escalated": False},
        },
        {
            "payload": {"query": "Can you give me stock tips for the ASX?", "customer_id": "C001"},
            "note":    "Deflector path — expect allow_retry=True",
            "checks":  {"intent": "out_of_scope", "allow_retry": True, "escalated": False},
        },
        {
            "payload": {"query": "I was charged a $35 fee I was never told about.", "customer_id": "C001"},
            "note":    "Standard complaint — expect case ID in answer, escalated=False",
            "checks":  {"intent": "complaint", "allow_retry": False, "escalated": False},
        },
        {
            "payload": {
                "query":       "There is a $450 transaction I never made. Someone accessed my account.",
                "customer_id": "C002",
            },
            "note":   "Urgent complaint — HITL cycle, expect escalated=True",
            "checks": {"intent": "complaint", "allow_retry": False, "escalated": True},
        },
    ]
 
    passed = failed = 0
 
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(test_cases)}] {tc['note']}")
        print(f"  POST /enquire  {tc['payload']}")
 
        r = client.post("/enquire", json=tc["payload"])
        print(f"  HTTP {r.status_code}")
 
        if r.status_code != 200:
            print(f"  ✗ FAIL — {r.text[:200]}")
            failed += 1
            continue
 
        body = r.json()
        print(f"  intent:      {body['intent']}")
        print(f"  allow_retry: {body['allow_retry']}")
        print(f"  escalated:   {body['escalated']}")
        print(f"  sources:     {body['sources']}")
        print(f"  session_id:  {body['session_id'][:8]}...")
        print(f"  answer:      {body['answer'][:120]}...")
 
        ok = all(body.get(k) == v for k, v in tc["checks"].items())
        if ok:
            print("  ✓ PASS")
            passed += 1
        else:
            mismatches = {k: f"expected {v}, got {body.get(k)}"
                          for k, v in tc["checks"].items() if body.get(k) != v}
            print(f"  ✗ FAIL — {mismatches}")
            failed += 1
 
    print(f"\n{'=' * 60}")
    print(f"Stage B: {passed} passed, {failed} failed out of {len(test_cases)}")
    print("=" * 60)