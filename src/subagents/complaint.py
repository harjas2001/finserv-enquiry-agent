"""
Phase 3 — Complaint Handler Subagent
src/subagents/complaint.py
=============================
Handles customer complaints and disputes. Logs every complaint to a mock case
management system (data/complaints.json), classifies severity, and for
urgent cases flags the enquiry for human-in-the-loop (HITL) review before
the response is sent.
 
Graph position:
    orchestrator_node → (intent="complaint") → complaint_node
                                                     │
                                          route_after_complaint()
                                                     │
                              ┌──────────────────────┴───────────────────────┐
                              │                                                │
                       "escalate"                                       "continue"
                              │                                                │
                              ▼                                                ▼
                      escalation_node                                  guardrail_node
                       (interrupt — HITL)                                      │
                              │                                                │
                              └──────────────────────► guardrail_node ◄────────┘
 
Reads from state:   query, customer_id
Writes to state:    subagent_response, sources (always []), escalated
 
 
log_complaint is a WRITE:
it creates a new case record with a side effect (a row appended to
data/complaints.json, a case ID generated). This is the same shape as a real
"create ticket" API call in a case management system (Zendesk, Salesforce
Service Cloud, etc.) — the mock just uses a JSON file instead of a database.
 
Same two-turn pattern as account.py:
  Turn 1 (forced, mode="ANY") → Gemini extracts a complaint summary, category,
                                  and severity, and calls log_complaint
  Execute                     → write the case record, return a case_id
  Turn 2                      → Gemini composes the acknowledgement to the
                                  customer using the case_id and severity
 
Defense-in-depth severity check
──────────────────────────────────
The account subagent investigation found that Gemini will fill a required
schema field with a plausible-but-wrong value when it has no real information
(it invented a customer_id). The same risk applies here to `severity` — it's
a safety-critical field (it decides whether a human reviews the case), and an
LLM misclassification ("standard" when it should be "urgent") would mean a
fraud complaint goes straight back to the customer with no human review.
 
The fix here is different from account.py (we can't remove `severity` from
the schema — Gemini's classification IS the point) but the principle is the
same: don't let the LLM be the only check on a safety-critical decision.
_apply_severity_override() scans the complaint text for fraud-related
keywords and forces severity="urgent" regardless of Gemini's classification
if any are found. Gemini's classification is the primary signal; the keyword
check is the backstop.
 
HITL escalation — why interrupt() can't be tested standalone
────────────────────────────────────────────────────────────
LangGraph's interrupt() pauses execution of a *compiled graph* and requires a
checkpointer (e.g. MemorySaver) to persist state across the pause. Calling it
outside that context raises GraphInterrupt with nowhere to resume from.
 
complaint_node and route_after_complaint are fully testable standalone (this
file's test harness does so). escalation_node is defined here because it's
complaint-specific, but it can only be exercised once src/graph.py wires it
into the compiled graph with add_conditional_edges() + a checkpointer —
the final step of Phase 3. Today's test harness checks it's structurally
correct and documents the deferred test.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
 
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.types import interrupt
 
from src.state import EnquiryState
 
load_dotenv()


#CONFIG
GEMINI_MODEL = "gemini-2.5-flash"
 
# Resolved relative to this file: src/subagents/complaint.py
COMPLAINTS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "complaints.json"

# Defence-in-depth severity override.
# List below of key words will triggers an instant HITL.
# "severity" will be assinged "urgent" and overwrites 
# geminis classification. Important to note the list should be company client specific
# meaning after analysing key escalstion queries then extracting top "escalated" words
# not just adding every potential word, as that could cause unneccary contact 
# centre overload.
FRAUD_KEYWORDS = [
    "fraud", "scam", "scammed", "stolen", "hacked",
    "unauthorised", "unauthorized",
    "didn't authorise", "didn't authorize",
    "identity theft", "without my consent", "without my permission",
]

SYSTEM_PROMPT = """\
You are a complaints specialist for Clearwater Bank.
 
Your role is to acknowledge customer complaints professionally and
empathetically, log every complaint accurately in the case management
system, and explain next steps.
 
Rules:
1. Always log every complaint using the log_complaint tool, never skip this,
   even if the complaint seems minor or vague.
2. Write the "description" argument as a concise, factual, third-person
   summary. Example: "Customer reports an unauthorised transaction of $450
   on their everyday account."
3. Classify "category" as one of: fees, service, transaction_dispute, fraud, other.
4. Classify "severity" as "urgent" if the complaint involves suspected fraud,
   unauthorised transactions, identity theft, threats of legal action, or
   significant financial harm. Otherwise use "standard".
5. In your response to the customer:
   - Acknowledge their concern without admitting fault, assigning blame, or
     promising a specific outcome, refund, or compensation amount.
   - Always provide the case reference number from the tool result.
   - For "standard" severity, tell them to expect a response within 5
     business days.
   - For "urgent" severity, tell them their case has been escalated for
     priority review and they may be contacted sooner.
6. Keep responses concise and professional.\
"""

