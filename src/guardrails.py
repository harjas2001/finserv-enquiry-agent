"""
Phase 4 — Guardrail Node
src/guardrails.py
==================
Every subagent_response flows through this node before becoming final_response.
Three sequential checks protect against the most critical failure modes in a
regulated financial services context.
 
Check 1 — PII detector (hard block)
    Regex scan for Australian BSB and account number formats.
    If detected: redact PII information and retunr final_message.
    Connection to account.py: the account subagent already masks account numbers
    to last-4 digits at the tool level. This is defence-in-depth, a second line
    of protection in case masking is bypassed or a prompt injection occurs.
 
Check 2 — Hallucination risk (soft flag, RAG responses only)
    When sources is non-empty (product subagent ran RAG), verify that key terms
    from the retrieved chunks appear in the response.
    If few terms overlap: set guardrail_flags["hallucination_risk"] = True.
    Soft flag: response is still served. The eval harness applies LLM-as-judge
    scoring offline where this flag triggers closer review.
 
    Architecture note: keeping this as a lightweight overlap check at runtime
    (rather than a Gemini API call) is a deliberate design choice. Runtime
    guardrails should be fast and cheap. LLM-as-judge belongs in the eval
    harness (offline, asynchronous). This separation is a portfolio talking point.
 
Check 3 — Scope violation (hard block)
    Regex scan for financial advice language ("I recommend", "you should invest").
    If detected: replace response with a regulatory-safe refusal.
    Connection to deflector.py: the deflector handles out-of-scope routing at the
    orchestrator level. This catches cases where an in-scope subagent (e.g. product)
    drifts into giving investment recommendations.
 
Node signature:
    Reads from state:  subagent_response, sources, retrieved_chunks
    Writes to state:   final_response, guardrail_flags
"""

import re

from src.state import EnquiryState


#PII PATTERNS
#Aus specific formats...
#Implementing two PII identifiers, the two major ones for banks/financial services
# BSB (Bank State Branch): the 6-digit routing code used in every AU bank transfer.
# Format: XXX-XXX (hyphen required — bare 6-digit numbers are too common in
# other contexts to flag indiscriminately, e.g. postcodes chained together).
#
# Account numbers: 6–10 contiguous digits. Negative lookbehind on $ excludes
# dollar amounts (e.g. $123456.00). Word boundaries (\b) prevent matching
# substrings inside longer digit strings — phone numbers formatted with spaces
# ("1300 555 100") are safe because the spaces break the word boundary.

_BSB_RE = re.compile(
    r'\b\d{3}-\d{3}\b'
)
_ACC_RE = re.compile(
    r'(?<!\$)(?<!\d)\b\d{6,10}\b(?!\d)'
)


#FINANCIAL ADVICE PHRASES
# In regulated Australian financial services (ASIC), providing personal financial
# advice without a licence is a legal liability. These patterns are conservative:
# factual product info ("the variable rate is 6.54% p.a.") is fine;
# directed recommendations ("you should take the fixed rate") are not.

#Need to apply the IGNORECASE flag as Geminis output is non-deterministic
_ADVICE_PATTERNS =[
    re.compile(r, re.IGNORECASE) for r in [
        r'\byou should (?:invest|buy|sell|put|choose|take|switch|consider)\b',
        r'\bI (?:recommend|suggest|advise)\b',
        r'\bI\'d recommend\b',
        r'\bwould recommend\b',
        r'\bbest (?:investment|option|choice|product) for you\b',
        r'\bpersonally suggest\b',
        r'\bconsider (?:investing|buying|selling|switching)\b',
        r'\bmy advice (?:would be|is)\b',        
    ]
]


#FALLBACK MESSAGES
_PII_FALLBACK = (
    "For security reasons, I'm unable to display that account information here. "
    "Please log in to Clearwater Online Banking or call us on 1300 555 100 "
    "to review your account details securely."
)

_SCOPE_FALLBACK = (
    "I'm not able to provide personal financial advice. For tailored guidance, "
    "please speak with a licensed Clearwater financial adviser by calling "
    "1300 555 100 or visiting your nearest branch."   
)


#STOPWORDS FOR HALLUCINATION CHECK
# Excluded from the "key chunk terms" set used in the overlap check.
# The goal is to keep only domain-specific terms (rates, product features,
# financial jargon) that would only appear in a genuinely grounded response.
# Generic words appear in both grounded and hallucinated responses, they
# dilute the overlap signal rather than strengthen it.
_STOPWORDS = {
    "about", "after", "also", "back", "bank", "been", "before", "between",
    "both", "clearwater", "could", "does", "each", "even", "from", "have",
    "here", "into", "just", "keep", "know", "like", "make", "more", "most",
    "much", "need", "only", "other", "over", "please", "should", "some",
    "such", "than", "that", "their", "them", "then", "there", "these", "they",
    "this", "those", "through", "time", "under", "upon", "used", "very",
    "want", "well", "were", "what", "when", "which", "will", "with", "within",
    "without", "would", "your",
}



#CHECK 1: PII DETECTOR
def _check_pii(text: str) -> tuple[bool, str]:
    #Scan for Aus banking PII patterns
    if _BSB_RE.search(text):
        return True, "BSB number pattern detected"
    if _ACC_RE.search(text):
        return True, "account number pattern detected"
    return False, ""

def _redact_pii(text: str) -> str:
    # Redact detected PII fragments in-place.
    # Called only when _check_pii() has already confirmed a match, so we know
    # at least one pattern will fire.
    text = _BSB_RE.sub("[BSB REDACTED]", text)
    text = _ACC_RE.sub(lambda m: f"****{m.group()[-4:]}", text)     # only shows last 4 digits of account number similar to account subagent

    return text


#CHECK 2: HALLUCINATION RISK (RAG path only)
def _check_hallucination(response: str, chunks: list[str]) -> bool:
    """
    Term overlap check between retrieved chunks and the generated response.
 
    Returns True (hallucination risk) if the response shares fewer than 15%
    of key terms with the source chunks.
 
    Why 15%?
        The threshold is deliberately lenient. Gemini paraphrases heavily, so
        exact term matching will always undercount true grounding. 15% means
        roughly 1 in 7 key chunk words must appear, a clear signal the LLM
        engaged with the retrieved content rather than ignoring it.
 
    Why not LLM-as-judge here?
        Runtime guardrails must be fast and cheap, every enquiry passes through
        this node synchronously. An extra Gemini call adds ~1s latency and cost
        per request. LLM-as-judge is applied offline in run_evals.py (the eval
        harness) where latency is not a constraint. Separating runtime trip-wire
        from offline eval scoring is a deliberate architecture choice.
    """
    #None RAG paths
    if not chunks:
        return False
    
    chunk_text      = " ".join(chunks).lower()
    response_text   = response.lower()
    # Extract key terms: words longer than 4 chars, not in stopword list.
    chunk_terms = {
        word for word in re.findall(r'\b[a-z]{5,}\b', chunk_text)
        if word not in _STOPWORDS
    }

    #no meaningful terms found so skip check
    if not chunk_terms:
        return False
    
    matched_count = sum(1 for term in chunk_terms if term in response_text)
    overlap_ratio = matched_count / len(chunk_terms)

    # True = hallucination risk flagged.
    return overlap_ratio < 0.15


#CHECK 3: SCOPE VIOLATION
def _check_scope(text: str) -> bool:
    # Detect financial advice lanaguage in response
    return any(pattern.search(text) for pattern in _ADVICE_PATTERNS)


#GUARDRAIL NODE
def guardrail_node(state: EnquiryState) -> dict:
    """
    LangGraph node: applies all three checks and writes final_response.
 
    Graph position:
        [account | product | deflector | escalation] → guardrail_node → END
 
    Reads from state:
        subagent_response  — raw answer from whichever subagent ran
        sources            — source filenames (non-empty = RAG path)
        retrieved_chunks   — raw chunk texts (populated when sources is non-empty)
 
    Writes to state:
        final_response     — guardrail-cleared answer sent to the customer
        guardrail_flags    — dict of all check results (for logging and evals)
 
    Check priority:
        All three checks run against the ORIGINAL subagent_response (not the
        fallback). This ensures a complete picture of what the subagent produced,
        even when the response gets replaced.
 
        Hard blocks (replace final_response):
            Scope check runs first  → _SCOPE_FALLBACK
            PII check runs second   → _PII_FALLBACK  (overrides scope if both fire)
 
        Soft flag (response still served):
            Hallucination check     → guardrail_flags["hallucination_risk"] = True
 
        PII overrides scope because leaking account numbers is a higher-severity
        compliance event than generating advice language. Both firing simultaneously
        is extremely unlikely in practice.
    """
    
    response    = state["subagent_response"]
    sources     = state["sources"]
    chunks      = state.get("retrieved_chunks", [])

    flags = {
        "pii_detected":         False,
        "hallucination_risk":   False,
        "out_of_scope":         False,
    }  

    #Default: pass subagent response as is
    final = response

    #In order of HIERARCHY
    #Check3: Scope
    if _check_scope(response):
        flags["out_of_scope"] = True
        final = _SCOPE_FALLBACK
        print("[GUARDRAIL] ⚠  Scope violation — financial advice language detected")
    
    #Check1: PII
    pii_flagged, pii_reason = _check_pii(response)
    if pii_flagged:
        flags["pii_detected"] = True,
        final = _redact_pii(final)
        print(f"[GUARDRAIL] ⚠  PII detected ({pii_reason}) — redacted in-place")

    #Check2: Hallucination
    if sources:
        if _check_hallucination(response, chunks):
            flags["hallucination_risk"] = True
            print("[GUARDRAIL] ⚠  Hallucination risk — low term overlap with retrieved chunks")

    
    #log clean pass
    if not any(flags.values()):
        print("[GUARDRAIL] ✓  All checks passed — response cleared")

    return {
        "final_response":   final,
        "guardrail_flags":  flags,
    }


# STAGE A TEST HARNESS
# Tests each private checker function independently (pure Python — no LangGraph,
# no Gemini). Same Stage A pattern used throughout Phase 3.
# Stage B is python -m src.graph.
 
if __name__ == "__main__":
    import sys
 
    print("=" * 60)
    print("GUARDRAIL — Stage A test harness (pure Python)")
    print("=" * 60)
 
    failures = []
 
    # ── PII tests ─────────────────────────────────────────────────────────────
    print("\n[1] PII detector")
 
    
 
    # ── Scope tests ───────────────────────────────────────────────────────────
    print("\n[2] Scope check")
 
    pii_cases = [
        # (text, should_flag, expected_output_contains, label)
        (
            "Your BSB is 012-003 and your account is ready.",
            True, "[BSB REDACTED]",
            "BSB with hyphen",
        ),
        (
            "Your account number is 123456789.",
            True, "****6789",
            "9-digit account number → masked to last 4",
        ),
        (
            "Your balance is $4,823.17.",
            False, "$4,823.17",
            "Dollar amount — not PII",
        ),
        (
            "Call us on 1300 555 100 for assistance.",
            False, "1300 555 100",
            "Phone with spaces — not PII",
        ),
        (
            "No sensitive data in this response.",
            False, "No sensitive data",
            "Clean response",
        ),
    ]

    for text, should_flag, expected_fragment, label in pii_cases:
        flagged, reason = _check_pii(text)
        output = _redact_pii(text) if flagged else text
        ok = (flagged == should_flag) and (expected_fragment in output)
        if not ok:
            failures.append(f"PII [{label}]: flagged={flagged}, output={output!r}")
        detail = f"→ {output!r}" if flagged else ""
        print(f"  {'✓' if ok else '✗'} {label}: flagged={flagged} {detail}")
 
    # ── Hallucination tests ───────────────────────────────────────────────────
    print("\n[3] Hallucination check")
 
    sample_chunk = (
        "The variable interest rate on our ClearHome Standard home loan is 6.54% p.a. "
        "comparison rate 6.78% including offset account features and redraw facility."
    )
    grounded     = (
        "Clearwater's variable rate home loan is currently 6.54% p.a. with a comparison "
        "rate of 6.78%. Features include an offset account and redraw facility."
    )
    hallucinated = (
        "Our home loan rates are highly competitive. We offer some of the best rates in "
        "the market with flexible repayment options to suit your individual needs."
    )
 
    hal_cases = [
        (grounded,     [sample_chunk], False, "Grounded response — should NOT flag"),
        (hallucinated, [sample_chunk], True,  "Generic response — should flag"),
        (grounded,     [],             False, "No chunks (non-RAG) — should NOT flag"),
    ]
 
    for response, chunks, should_flag, label in hal_cases:
        flagged = _check_hallucination(response, chunks)
        ok = flagged == should_flag
        if not ok:
            failures.append(f"Hallucination [{label}]: expected {should_flag}, got {flagged}")
        print(f"  {'✓' if ok else '✗'} {label}: flagged={flagged}")
 
    # ── guardrail_node integration ────────────────────────────────────────────
    print("\n[4] guardrail_node — mock state (clean response)")
 
    from src.state import make_initial_state
 
    mock_state = make_initial_state("What is the home loan rate?")
    mock_state["subagent_response"] = "The variable rate is 6.54% p.a. with an offset account facility."
    mock_state["sources"]           = ["home_loan_guide.pdf"]
    mock_state["retrieved_chunks"]  = [sample_chunk]
 
    result = guardrail_node(mock_state)
    node_ok = (
        result["final_response"]              == mock_state["subagent_response"]
        and result["guardrail_flags"]["pii_detected"]       == False
        and result["guardrail_flags"]["hallucination_risk"] == False
        and result["guardrail_flags"]["out_of_scope"]       == False
    )
    if not node_ok:
        failures.append("guardrail_node: clean response should pass all checks unchanged")
    print(f"  {'✓' if node_ok else '✗'} Clean RAG response passes all checks")
    print(f"    final_response  = {result['final_response']!r}")
    print(f"    guardrail_flags = {result['guardrail_flags']}")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failures:
        print(f"✗ {len(failures)} test(s) FAILED:")
        for f in failures:
            print(f"  • {f}")
        sys.exit(1)
    else:
        print("✓ All Stage A tests passed.")
    print("=" * 60)