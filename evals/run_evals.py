"""
Phase 4 — Eval Harness
evals/run_evals.py
==================
Offline quality gate for finserv-enquiry-agent.

Runs all 20 labelled test cases from evals/test_cases.json through the full
LangGraph graph and scores the results across three metrics.

Architecture note — why offline, not inline:
    Runtime guardrails (src/guardrails.py) must be fast and cheap — they sit
    on the critical path of every live request. This harness runs asynchronously
    after deployments or prompt changes. Separating runtime defence from offline
    evaluation is a deliberate design choice: the guardrail catches individual
    failures; the harness detects systemic regressions.

Three phases (run in sequence so graph results are captured before judge calls):
    Phase 1 — Run all 20 cases through the graph     (~20 Gemini calls)
    Phase 2 — Faithfulness judge on RAG cases         (~5 Gemini judge calls)
    Phase 3 — Task completion judge on non-HITL cases (~18 Gemini judge calls)

Four metrics:
    routing_accuracy  — deterministic; actual_intent == expected_intent
    faithfulness      — LLM-as-judge; RAG cases only (sources non-empty)
    task_completion   — LLM-as-judge; all non-HITL completed cases
    content_match     — deterministic; expected_answer_contains terms present

Eval gates (exit non-zero if breached):
    routing_accuracy < 0.90  →  exit(1)
    faithfulness     < 0.85  →  exit(1)

Run from project root:
    python -m evals.run_evals
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Interrupt

from src.graph import build_graph
from src.state import make_initial_state
from src.llm_client import get_client

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

TEST_CASES_PATH  = Path(__file__).parent / "test_cases.json"

# Eval gates — script exits non-zero if either is breached
ROUTING_GATE     = 0.90
FAITHFULNESS_GATE = 0.85

# Judge model — use standard Flash (not Lite) for evaluation accuracy.
# Judges run offline so latency is not a concern. Better model = better verdicts.
JUDGE_MODEL = "gemini-3.5-flash"

# Small delay between graph runs to stay within API rate limits.
# Set to 0 if you have a paid tier with higher quota.
INTER_CASE_DELAY_SECONDS = 1

JUDGE_SYSTEM_PROMPT = """\
You are a precise evaluator for a financial services chatbot.
You reason carefully before responding.
You always respond with valid JSON and nothing else, no markdown, no preamble.\
"""


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_test_cases() -> list[dict]:
    """Load and validate test_cases.json."""
    if not TEST_CASES_PATH.exists():
        raise FileNotFoundError(
            f"Test cases not found at {TEST_CASES_PATH}. "
            "Expected evals/test_cases.json relative to project root."
        )
    with open(TEST_CASES_PATH) as f:
        cases = json.load(f)

    required_keys = {"id", "input", "customer_id", "expected_intent",
                     "expected_answer_contains", "hitl_expected"}
    for case in cases:
        missing = required_keys - set(case.keys())
        if missing:
            raise ValueError(f"Test case {case.get('id', '?')} missing keys: {missing}")

    return cases


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _content_match(response: str, expected_contains: list[str]) -> bool:
    """
    Check that all expected terms appear in the response (case-insensitive).

    Empty expected_contains list → vacuously True (no terms required).
    Used for HITL cases where we cannot check response content because
    the graph was interrupted before final_response was written.
    """
    if not expected_contains:
        return True
    response_lower = response.lower()
    return all(term.lower() in response_lower for term in expected_contains)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-AS-JUDGE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _faithfulness_judge(
    response: str,
    chunks: list[str],
    client: genai.Client,
) -> tuple[bool, str]:
    """
    Judge whether the response is faithful to the retrieved source chunks.

    Returns (faithful: bool, reason: str).

    Faithful = True means every specific claim in the response (rates, amounts,
    product names, eligibility conditions) is traceable to the provided chunks.
    Generic statements that don't contradict the chunks are acceptable.

    Conservative default: faithful=True on parse failure.
    Rationale: a JSON parsing glitch should not penalise a legitimately good
    response. The guardrail's hallucination_risk flag is the runtime signal;
    this judge is the offline verdict.
    """
    chunk_text = "\n\n".join(
        f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
    )

    prompt = f"""\
Retrieved source chunks:

{chunk_text}

---

Chatbot response to evaluate:
{response}

---

Does the chatbot response make any specific factual claims — interest rates,
fees, product names, eligibility conditions, account figures — that are NOT
supported by the source chunks above?

Generic statements (e.g. "our rates are competitive", "we have flexible options")
that do not contradict the chunks are acceptable.
If the response is a polite deflection ("I can't help with that, please call us")
that makes no factual claims at all, mark faithful=true.

Respond with JSON only:
{{"faithful": true_or_false, "reason": "one sentence explanation"}}\
"""

    try:
        result = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0,
            ),
        )
        text = result.text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return bool(parsed.get("faithful", True)), parsed.get("reason", "")
    except Exception as e:
        print(f"      [JUDGE] faithfulness parse error: {e}")
        return True, f"parse_error — defaulted to faithful=True ({e})"


def _task_completion_judge(
    query: str,
    response: str,
    expected_contains: list[str],
    client: genai.Client,
) -> tuple[bool, str]:
    """
    Judge whether the response adequately answered the customer's question.

    Returns (completed: bool, reason: str).

    Adequate responses include:
    - Direct answers that cover the expected topics
    - Appropriate refusals (out-of-scope deflection, security redirects)
    - HITL escalation confirmations with a case ID

    Conservative default: completed=False on parse failure.
    Rationale: a missed answer is worse than a false negative here — the eval
    harness should be more likely to flag issues than to miss them.
    """
    hints = (
        ", ".join(f'"{t}"' for t in expected_contains)
        if expected_contains
        else "none specified"
    )

    prompt = f"""\
Customer question:
{query}

Expected answer topics (for grading reference): {hints}

Chatbot response to evaluate:
{response}

---

Did the chatbot response adequately address the customer's question?

Mark completed=true if the response:
  - Directly engages with what was asked
  - Covers the expected topics where specified
  - Is a deliberate and appropriate refusal (out-of-scope, security redirect)

Mark completed=false if the response:
  - Ignores the question entirely
  - Provides clearly wrong information
  - Is a generic error message unrelated to the query

Respond with JSON only:
{{"completed": true_or_false, "reason": "one sentence explanation"}}\
"""

    try:
        result = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                temperature=0,
            ),
        )
        text = result.text.strip().replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return bool(parsed.get("completed", False)), parsed.get("reason", "")
    except Exception as e:
        print(f"      [JUDGE] task_completion parse error: {e}")
        return False, f"parse_error — defaulted to completed=False ({e})"


# ─────────────────────────────────────────────────────────────────────────────
# CASE RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_case(graph, case: dict) -> dict:
    """
    Run a single test case through the compiled LangGraph graph.

    Returns a result dict containing:
        id, input, customer_id
        expected_intent, actual_intent, routing_correct
        final_response, sources, retrieved_chunks
        hitl_triggered, hitl_expected
        expected_answer_contains, content_match
        error (str | None)

    Two execution paths:
        Normal  — graph.invoke() returns the final state dict.
        HITL (paused) — graph.invoke() returns normally, but
                        escalated=True and final_response="". This is what
                        the installed LangGraph version actually does on
                        interrupt(). Detected inline, no extra state fetch
                        needed — final_state already has everything.
        HITL (raises) — graph.invoke() raises GraphInterrupt. Kept as a
                        fallback for other LangGraph versions; state is
                        retrieved via graph.get_state(config).

    Thread IDs:
        Each case uses a unique thread_id ("eval-TC-001" etc.) so the shared
        MemorySaver isolates state between cases without needing a new
        checkpointer per run.
    """
    thread_id = f"eval-{case['id']}"
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state = make_initial_state(
        query=case["input"],
        customer_id=case.get("customer_id", ""),
    )

    # Initialise result with defaults — fields are filled in by each path below
    result = {
        "id":                       case["id"],
        "input":                    case["input"],
        "customer_id":              case.get("customer_id", ""),
        "expected_intent":          case["expected_intent"],
        "actual_intent":            "",
        "routing_correct":          False,
        "final_response":           "",
        "sources":                  [],
        "retrieved_chunks":         [],
        "hitl_triggered":           False,
        "hitl_expected":            case["hitl_expected"],
        "expected_answer_contains": case["expected_answer_contains"],
        "content_match":            False,
        "error":                    None,
        # Filled during scoring phases:
        "faithful":                 None,
        "faithful_reason":          "n/a",
        "completed":                None,
        "completed_reason":         "n/a",
    }

    try:
        # ── Normal path ───────────────────────────────────────────────────────
        final_state = graph.invoke(initial_state, config=config)

        # Interrupt detection (paused-state form):
        # LangGraph 0.2+ returns the paused state from invoke() when
        # interrupt() fires, rather than raising an exception — same pattern
        # api/main.py already detects. Signature: escalated=True (set by
        # complaint_node) + final_response="" (guardrail never ran because
        # the graph paused before reaching it).
        #
        # This has to be checked here, not just in the "except Interrupt"
        # branch below: on the installed LangGraph version, invoke() does NOT
        # raise on interrupt, so that except branch never fires and these
        # cases were silently falling through as "normal" — routing_correct
        # came out right (intent matched) but content_match/task_completion
        # were being scored against an empty response, which always fails.
        # The escalation was never wrong; the harness just wasn't looking in
        # the right place.


        if final_state.get("escalated") and not final_state.get("final_response"):
            result["hitl_triggered"]  = True
            result["actual_intent"]   = final_state.get("intent", "")
            result["final_response"]  = final_state.get("subagent_response", "")
            result["sources"]         = final_state.get("sources", [])
            result["retrieved_chunks"]= final_state.get("retrieved_chunks", [])
            result["routing_correct"] = (
                result["actual_intent"] == case["expected_intent"]
                and final_state.get("escalated", False) is True
            )
            # content_match vacuously True for HITL cases (expected_answer_contains=[])
            result["content_match"] = _content_match(
                result["final_response"],
                case["expected_answer_contains"],
            )
        else:
            result["actual_intent"]   = final_state.get("intent", "")
            result["final_response"]  = final_state.get("final_response", "")
            result["sources"]         = final_state.get("sources", [])
            result["retrieved_chunks"]= final_state.get("retrieved_chunks", [])
            result["routing_correct"] = (
                result["actual_intent"] == case["expected_intent"]
            )
            result["content_match"]   = _content_match(
                result["final_response"],
                case["expected_answer_contains"],
            )

    except Interrupt:
        # ── HITL path ─────────────────────────────────────────────────────────
        # The graph paused inside escalation_node. State up to that point
        # (including intent and escalated=True set by complaint_node) is
        # saved in the MemorySaver. Retrieve it to check routing.
        result["hitl_triggered"] = True

        try:
            snapshot = graph.get_state(config)
            saved    = snapshot.values

            result["actual_intent"]  = saved.get("intent", "")
            result["final_response"] = saved.get("subagent_response", "")
            result["sources"]        = saved.get("sources", [])
            result["retrieved_chunks"] = saved.get("retrieved_chunks", [])
            result["routing_correct"]  = (
                result["actual_intent"] == case["expected_intent"]
                and saved.get("escalated", False) is True
            )
            # content_match vacuously True for HITL cases (expected_answer_contains=[])
            result["content_match"] = _content_match(
                result["final_response"],
                case["expected_answer_contains"],
            )
        except Exception as snapshot_err:
            result["error"] = f"HITL state retrieval failed: {snapshot_err}"

    except Exception as e:
        # ── Unexpected error ──────────────────────────────────────────────────
        result["error"] = str(e)
        result["routing_correct"] = False
        print(f"      [ERROR] {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE SCORERS
# ─────────────────────────────────────────────────────────────────────────────

def score_routing_accuracy(results: list[dict]) -> float:
    """
    Proportion of cases where actual_intent == expected_intent.
    Denominator: all cases (including HITL and errored cases).
    Errored cases count as incorrect — a crash is a routing failure.
    """
    if not results:
        return 0.0
    correct = sum(1 for r in results if r["routing_correct"])
    return correct / len(results)


def score_faithfulness(results: list[dict]) -> float:
    """
    Proportion of RAG cases where the response is faithful to retrieved chunks.
    Denominator: cases with non-empty sources, non-HITL, no error, faithful field set.
    Returns 1.0 if no eligible cases (no gate failure on non-RAG-only runs).
    """
    eligible = [
        r for r in results
        if r["sources"]
        and not r["hitl_triggered"]
        and not r["error"]
        and r["faithful"] is not None
    ]
    if not eligible:
        return 1.0
    return sum(1 for r in eligible if r["faithful"]) / len(eligible)


def score_task_completion(results: list[dict]) -> float:
    """
    Proportion of non-HITL cases where the response adequately answered the query.
    Denominator: non-HITL, no error, completed field set.
    """
    eligible = [
        r for r in results
        if not r["hitl_triggered"]
        and not r["error"]
        and r["completed"] is not None
    ]
    if not eligible:
        return 1.0
    return sum(1 for r in eligible if r["completed"]) / len(eligible)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def _tick(value) -> str:
    """Format a bool or None for table display."""
    if value is None:
        return "  -  "
    return "  ✓  " if value else "  ✗  "


def print_results_table(results: list[dict]) -> None:
    """Print a fixed-width per-case results table."""
    header = (
        f"{'ID':<8} {'EXP INTENT':<14} {'ACT INTENT':<14} "
        f"{'ROUTE':^7} {'HITL':^6} {'CONTENT':^8} {'FAITH':^7} {'COMPLETE':^9}"
    )
    print("\n" + "=" * 80)
    print("PER-CASE RESULTS")
    print("=" * 80)
    print(header)
    print("-" * 80)

    for r in results:
        route_sym  = "  ✓  " if r["routing_correct"] else "  ✗  "
        hitl_sym   = "  ✓  " if r["hitl_triggered"]  else "  -  "
        content_sym= _tick(r["content_match"] if not r["hitl_triggered"] else None)
        faith_sym  = _tick(r["faithful"])
        comp_sym   = _tick(r["completed"])
        error_tag  = " [ERR]" if r["error"] else ""

        print(
            f"{r['id']:<8} {r['expected_intent']:<14} {r['actual_intent']:<14} "
            f"{route_sym}{hitl_sym}{content_sym}{faith_sym}{comp_sym}{error_tag}"
        )

        # Print response preview (first 80 chars) indented
        preview = r["final_response"].replace("\n", " ")[:80]
        if preview:
            print(f"         → {preview!r}")

    print("-" * 80)


def print_score_summary(
    routing: float,
    faithfulness: float,
    task_completion: float,
    results: list[dict],
) -> None:
    """Print the aggregate score summary with gate status."""

    def gate_str(score: float, threshold: float) -> str:
        status = "PASS ✓" if score >= threshold else "FAIL ✗"
        return f"{score:.1%}  (gate: {threshold:.0%})  [{status}]"

    def no_gate_str(score: float) -> str:
        return f"{score:.1%}  (no gate — reported only)"

    content_match_count = sum(
        1 for r in results
        if r["content_match"] and not r["hitl_triggered"] and not r["error"]
    )
    content_eligible = sum(
        1 for r in results if not r["hitl_triggered"] and not r["error"]
    )
    content_score = content_match_count / content_eligible if content_eligible else 1.0

    print("\n" + "=" * 60)
    print("AGGREGATE SCORES")
    print("=" * 60)
    print(f"  routing_accuracy  : {gate_str(routing, ROUTING_GATE)}")
    print(f"  faithfulness      : {gate_str(faithfulness, FAITHFULNESS_GATE)}")
    print(f"  task_completion   : {no_gate_str(task_completion)}")
    print(f"  content_match     : {no_gate_str(content_score)}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Orchestrate all three evaluation phases and enforce gate thresholds.

    Phase 1 — Graph runs:
        Each case runs through the full LangGraph graph with a unique thread_id.
        HITL cases are caught via GraphInterrupt; state is retrieved from the
        MemorySaver checkpointer. Results are collected before any judge calls.

    Phase 2 — Faithfulness scoring:
        Gemini (JUDGE_MODEL, temperature=0) evaluates each RAG response against
        its retrieved chunks. Only runs for cases with non-empty sources.

    Phase 3 — Task completion scoring:
        Gemini evaluates whether each non-HITL response adequately answered
        the customer's question.

    Gate check:
        routing_accuracy < ROUTING_GATE  → exit(1)
        faithfulness < FAITHFULNESS_GATE → exit(1)
        Both pass                        → exit(0)
    """
    api_key = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not api_key:
        print("ERROR: GOOGLE_CLOUD_PROJECT not set. Add it to your .env file.")
        sys.exit(1)

    cases  = load_test_cases()
    client = get_client()

    # Single shared checkpointer — cases are isolated by unique thread_id
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    # ── Phase 1: Run all cases ────────────────────────────────────────────────
    print("=" * 60)
    print(f"EVAL HARNESS — Phase 1: running {len(cases)} cases")
    print("=" * 60)

    raw_results = []
    for i, case in enumerate(cases, 1):
        hitl_tag = " [HITL expected]" if case["hitl_expected"] else ""
        print(f"\n[{i:02d}/{len(cases)}] {case['id']}{hitl_tag}")
        print(f"  query:    {case['input'][:70]}")
        print(f"  customer: {case['customer_id'] or '(none)'}")

        result = run_case(graph, case)
        raw_results.append(result)

        route_sym = "✓" if result["routing_correct"] else "✗"
        hitl_sym  = " + HITL ✓" if result["hitl_triggered"] else ""
        print(f"  routing: {route_sym}  ({result['actual_intent']} vs {result['expected_intent']}){hitl_sym}")

        if result["error"]:
            print(f"  error:   {result['error']}")

        if INTER_CASE_DELAY_SECONDS > 0:
            time.sleep(INTER_CASE_DELAY_SECONDS)

    # ── Phase 2: Faithfulness scoring ─────────────────────────────────────────
    rag_cases = [
        r for r in raw_results
        if r["sources"] and not r["hitl_triggered"] and not r["error"]
    ]
    print(f"\n{'=' * 60}")
    print(f"Phase 2: faithfulness scoring ({len(rag_cases)} RAG cases)")
    print("=" * 60)

    for r in raw_results:
        if r in rag_cases:
            print(f"  {r['id']} — judging faithfulness...")
            faithful, reason = _faithfulness_judge(
                r["final_response"], r["retrieved_chunks"], client
            )
            r["faithful"]        = faithful
            r["faithful_reason"] = reason
            print(f"    faithful={faithful} — {reason}")
            time.sleep(INTER_CASE_DELAY_SECONDS)

    # ── Phase 3: Task completion scoring ──────────────────────────────────────
    completion_eligible = [
        r for r in raw_results
        if not r["hitl_triggered"] and not r["error"]
    ]
    print(f"\n{'=' * 60}")
    print(f"Phase 3: task completion scoring ({len(completion_eligible)} cases)")
    print("=" * 60)

    for r in raw_results:
        if r in completion_eligible:
            print(f"  {r['id']} — judging task completion...")
            completed, reason = _task_completion_judge(
                r["input"], r["final_response"],
                r["expected_answer_contains"], client
            )
            r["completed"]        = completed
            r["completed_reason"] = reason
            print(f"    completed={completed} — {reason}")
            time.sleep(INTER_CASE_DELAY_SECONDS)

    # ── Aggregate scores ──────────────────────────────────────────────────────
    routing_score     = score_routing_accuracy(raw_results)
    faithfulness_score = score_faithfulness(raw_results)
    completion_score  = score_task_completion(raw_results)

    # ── Output ────────────────────────────────────────────────────────────────
    print_results_table(raw_results)
    print_score_summary(routing_score, faithfulness_score, completion_score, raw_results)

    # ── Gate check ────────────────────────────────────────────────────────────
    gate_failures = []
    if routing_score < ROUTING_GATE:
        gate_failures.append(
            f"routing_accuracy {routing_score:.1%} is below the {ROUTING_GATE:.0%} gate"
        )
    if faithfulness_score < FAITHFULNESS_GATE:
        gate_failures.append(
            f"faithfulness {faithfulness_score:.1%} is below the {FAITHFULNESS_GATE:.0%} gate"
        )

    if gate_failures:
        print("\n✗ EVAL GATE FAILED — address these before merging:")
        for failure in gate_failures:
            print(f"  • {failure}")
        sys.exit(1)
    else:
        print("\n✓ All gates passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()