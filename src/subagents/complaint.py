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
  Turn 1 (forced, mode="ANY") → `Gemini` extracts a complaint summary, category,
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


# CASE STORE
def _load_complaints() -> list:
    #Load the mock case list from disk. If [] then file is empty.
    if not COMPLAINTS_PATH.exists():
        return []
    with open(COMPLAINTS_PATH) as f:
        content = f.read().strip()
        return json.loads(content) if content else []
    
def _save_complaints(complaints: list) -> list:
    #Persist full case list back to disk
    with open(COMPLAINTS_PATH, "w") as f:
        json.dump(complaints, f, indent=2)


# SEVERITY OVERRIDE: defence-in-depth check
def _apply_severity_override(description: str, query: str, severity: str) -> str:
    #Return SEVERITY as "urgent" if geminis complaint summary or customer query
    #have a keyword in them already in the list.Its a deterministic sub system, 
    #to prevent LLM hallucination on this sensitive topic
    text = f"{description} {query}".lower()
    if any(keyword in text for keyword in FRAUD_KEYWORDS):
        return "urgent"
    return severity


#TOOL FUNCTION
def log_complaint(description: str, category: str, severity: str, customer_id: str) -> dict:
    """
    Creates a new complaint case record and persists it to data/complaints.json.
 
    WRITES a new record: A side effect with a generated case_id, the same shape 
    as calling a real case management API (Zendesk, Salesforce Service Cloud, etc.).
 
    customer_id is injected by complaint_node from session state, never
    supplied by Gemini. Same security principle as account.py: identity
    parameters are never part of the tool's schema (same for FunctionDeclaration
    below customer_id is not in `parameters`).
 
    Returns a dict with the case_id, status, category, and final severity —
    this is what Gemini sees in Turn 2 to compose its response.
    """
    complaints = _load_complaints()
    case_id = f"CW-2026-{len(complaints) + 1:04d}"

    record = {
        "case_id":          case_id,
        "customer_id":      customer_id,
        "category":         category,
        "severity":         severity,
        "description":      description,
        "status":           "open",
        "logged_at":        datetime.now().isoformat(timespec="seconds")
    }

    complaints.append(record)
    _save_complaints(complaints)

    return {
        "case_id":  case_id,
        "status":   "logged", #different to "open" one-time confirmation to gemini
        "category": category,
        "severity": severity,
    }


#FUNCTION DECLERATION
#customer_id cant be suuplied as a paramter value like other sub agents so gemini llm doesnt have any
#exposure and hallucinated risk towards it

_LOG_COMPLAINT_DECL = types.FunctionDeclaration(
    name="log_complaint",
    description=(
        "Logs a customer complaint into Clearwater Bank's case management "
        "system and returns a case reference number. Call this for every "
        "complaint, extract a clear, factual summary even if the customer's "
        "description is vague or emotional."
    ),
    parameters= {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": (
                    "A concise, factual, third-person summary of the complaint. "
                    "Example: 'Customer reports an unauthorised transaction of "
                    "$450 on their everyday account.'"
                ),
            },
            "category": {
                "type": "string",
                "enum": ["fees", "service", "transaction_dispute", "fraud", "other"],
                "description": "The category that best matches the complaint.",
            },
            "severity": {
                "type": "string",
                "enum": ["standard", "urgent"],
                "description": (
                    "'urgent' if the complaint involves suspected fraud, "
                    "unauthorised transactions, identity theft, threats of "
                    "legal action, or significant financial harm. 'standard' "
                    "for general service or fee complaints."
                ),
            },
        },
        "required": ["description", "category", "severity"],
    },
)

COMPLAINT_TOOLS = types.Tool(function_declarations=[_LOG_COMPLAINT_DECL])
DISPATCH: dict = {
    "log_complaint": log_complaint,
}


#COMPLAINT NODE

def complaint_node(state: EnquiryState) -> dict:
    """
    Main action point for complaint node:
    LangGraph node: logs the complaint via two-turn function calling and
    classifies its severity.
 
    Reads from state:  query, customer_id
    Writes to state:   subagent_response, sources, escalated
 
    escalated=True means route_after_complaint() will send this enquiry to
    escalation_node for human review before the guardrail layer.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError("GOOGLE_API_KEY is not set.")
 
    query       = state["query"]
    customer_id = state["customer_id"]
 
    print(f"\n[COMPLAINT] Query: '{query[:80]}'")
    print(f"[COMPLAINT] Customer ID: '{customer_id}'")

    try:
        client = genai.Client(api_key=api_key)

        # Turn 1: sending query + tool decleration to Gemini (forced)
        print("[COMPLAINT] Turn 1: sending query + tool declaration to Gemini (forced)")
        response1 = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[COMPLAINT_TOOLS],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                ),
                temperature=0,
            ),
        )

        function_call = None
        for part in response1.candidates[0].content.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                function_call = fc
                break

        #fallback incase no fc found which probably wont happen given tool_call is set to mode="ANY"
        #forcing Gemini to call a function
        if function_call is None:
            print("[COMPLAINT] Gemini answered directly (no tool call) — unexpected")
            return {
                "subagent_response": response1.text.strip(),
                "sources":           [],
                "escalated":         False,
            }
        

        tool_args = dict(function_call.args)
        gemini_severity = tool_args.get("severity", "standard")
        print(f"[COMPLAINT] Gemini classified: category='{tool_args.get('category')}', "
              f"severity='{gemini_severity}'")
        print(f"[COMPLAINT] Description: '{tool_args.get('description', '')}'")

        #Defence-in-depth severity overide
        final_severity = _apply_severity_override(
            description=str(tool_args.get("description", "")),
            query=query,
            severity=gemini_severity,
        )
        if final_severity != gemini_severity:
            print(f"[COMPLAINT] Keyword override — severity '{gemini_severity}' → "
                  f"'{final_severity}' (fraud-related language detected)")
        tool_args["severity"] = final_severity

        #Apply customer id to session statem no gemini involvement.
        tool_args["customer_id"] = customer_id  

        if function_call.name not in DISPATCH:
            raise ValueError(f"Unknown tool requested: '{function_call.name}'")      

        #Execute the tool: creating the case record in complaints.json
        tool_result = DISPATCH[function_call.name](**tool_args)
        print(f"[COMPLAINT] Case logged: {tool_result}")

        #TURN 2: customer facing acknowledgmement of complaint
        print("[COMPLAINT] Turn 2 — sending case result back to Gemini")
        response2 = client.models.generate_content(
            model=GEMINI_MODEL,
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text=query)],
                ),
                response1.candidates[0].content,
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=function_call.name,
                            response=tool_result,
                        )
                    )],
                ),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
            ),
        )

        final_answer = response2.text.strip()
        escalated = tool_result["severity"] == "urgent"
        print(f"[COMPLAINT] Final answer: '{final_answer[:100]}...'")
        print(f"[COMPLAINT] escalated = {escalated}")

        return {
            "subagent_response":    final_answer,
            "sources":              [],
            "escalated":            escalated,
        }




    except Exception as e:
        print(f"[COMPLAINT] ERROR: {type(e).__name__}: {e}")
        return {
            "subagent_response": (
                "I'm unable to log your complaint right now. Please call us "
                "on 1300 555 100 so we can assist you directly."
            ),
            "sources":   [],
            "escalated": False,
            "error":     f"Complaint subagent failed: {e}",
        }