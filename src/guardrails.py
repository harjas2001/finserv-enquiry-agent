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