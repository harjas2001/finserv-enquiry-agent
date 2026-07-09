"""
Phase 3 — Account Enquiry Subagent
src/subagents/account.py
============================
Handles customer queries about their own accounts — balances, transactions,
account history. Uses google-genai native function calling: Gemini decides
which tool to invoke, we execute it against mock_accounts.json, then Gemini
formats the result as a natural language response.
 
Graph position:
    orchestrator_node → (intent="account") → account_node → guardrail_node
 
Reads from state:   query, customer_id
Writes to state:    subagent_response, sources (always []), escalated (always False)
 
Tool calling — two turns explained
────────────────────────────────────
  Turn 1:  Send query + function declarations to Gemini.
           Gemini responds with a FunctionCall part — name + args.
 
  Execute: We call the Python function ourselves and get a result dict.
 
  Turn 2:  Send the original query + Gemini's function call + our result
           back to Gemini. Gemini now has the real data and composes a
           natural language answer.
 
Two API calls, one node, no LangGraph loop needed. The pattern is identical
 
Function declarations vs @tool
────────────────────────────────
In Phase 1, LangChain read @tool's docstring + type hints to generate the
tool schema Gemini sees. Here we write the schema directly as
types.FunctionDeclaration — JSON Schema for parameters, English description
for the "should I call this?" decision. More verbose, but nothing is hidden.
 
The DISPATCH dict
──────────────────
Gemini returns a tool call as a string name + args dict, e.g.:
    name = "get_account_balance", args = {"customer_id": "C001"}
DISPATCH["get_account_balance"](**args) maps that string to the real function.
This is the five-line version of what ToolNode did in Phase 1.

TO FIX:
- Have the customer id look up and error handling as a seperate code block so 
you dont always have to call the config.
"""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from google.genai import types

from src.state import EnquiryState
from src.llm_client import get_client

load_dotenv()


#CONFIG
GEMINI_MODEL = "gemini-2.5-flash-lite"
MOCK_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "mock_accounts.json"

SYSTEM_PROMPT = """\
You are an account specialist for Clearwater Bank. Your role is to help
customers with enquiries about their own accounts.
 
You have access to tools that retrieve live account data. Always use them,
never guess or fabricate balances, transaction amounts, or dates.
 
Rules:
1. Use a tool to retrieve account data before answering every query.
2. Present monetary figures with AUD currency and two decimal places.
3. For security, never state a full account number, show only the last 4 digits.
4. Keep responses concise, accurate, and professional.
5. Do not provide financial advice or investment recommendations.
6. If the requested information is not available, direct the customer to call
   1300 555 100 or visit a branch.
7. If a tool result contains an "error" field (for example, the customer is
   not identified), do not attempt to answer the original question. Instead,
   explain that you're unable to verify their identity and direct them to log
   in or call 1300 555 100. Never substitute a different customer's data.
8. "Last N transactions" (e.g. "last 3 transactions", "most recent 5") means
   a COUNT — use limit=N. Never use days=N for a count-based request.
   "Last N days" or "past N days" means a TIME WINDOW — use days=N.\
"""


#TOOL FUNCTION
def _load_mock_data() -> dict:
    #Load mock data on JSON file. Called on every tool invocation
    if not MOCK_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Mock data not found at {MOCK_DATA_PATH}. "
            "Check file path data/... and see if mock_accounts.json is there"
        )
    with open(MOCK_DATA_PATH) as f:
        return json.load(f)
    
def get_account_balance(customer_id: str) -> dict:
    # Returns current balances for all accounts belonging to the specific customer. 
    # Account numbers masked and only last 4 digits shown as normal practice for financial institutions

    data  = _load_mock_data()

    # In really authenticated app environment, customer is already logged in so more of a system 
    # lookup/ db failure.
    if not customer_id:
        return {"error": "No customer ID provided. Please log in to view account information."}
    
    if customer_id not in data:
        return {"error": f"No accounts found for customer '{customer_id}'. Please verify your details."}
    
    customer = data[customer_id]
    accounts = []
    for acc in customer["accounts"]:
        accounts.append({
            "account_type":             acc["account_type"],
            "account_number_last4":     acc["account_number"][-4:], # only last 4 digits
            "balance":                  acc["balance"],
            "currency":                 acc["currency"],
        })

    return {
        "customer_name":    customer["customer_name"],
        "accounts":         accounts,
    }

def get_recent_transactions(customer_id: str, days: int= 30, limit: int = None) -> dict:
    """
    Returns transactions from the last N days across all of the customer's accounts.
 
    Transactions are sorted by date descending (most recent first).
    The cutoff date is calculated from today's date at call time.
    
    limit (optional): if provided, caps the result to the N most recent
    transactions by count, applied after date filtering and sorting.
    Use when the customer asks for "last N transactions" as a count
    (e.g. "last 3 transactions" → limit=3), NOT for day-based queries.
    """

    data = _load_mock_data()

    if not customer_id:
        return {"error": "No customer ID provided. Please log in to view transaction history."}
    
    if customer_id not in data:
        return {"error": f"No accounts found for customer '{customer_id}'."}
    
    customer = data[customer_id]
    cutoff = date.today() - timedelta(days=days)

    all_transactions = []
    for acc in customer["accounts"]:
        for txn in acc.get("transactions", []):
            txn_date = date.fromisoformat(txn["date"])
            if txn_date >= cutoff:
                all_transactions.append({
                    "date":                     txn["date"],
                    "account_type":             acc["account_type"],
                    "account_number_last4":     acc["account_number"][-4:],
                    "description":              txn["description"],
                    "amount":                   txn["amount"],
                    "type":                     txn["type"], #credit or debit for example
                })
    
    #Get most recent transaction:
    all_transactions.sort(key=lambda x: x["date"], reverse=True)

    # Apply count limit if specified.
    # This is the fix for queries like "show me my last 3 transactions" —
    # limit=3 returns the 3 most recent entries regardless of their dates,
    # rather than filtering by the last 3 days (which may return nothing).
    if limit is not None:
        all_transactions = all_transactions[:limit]

    return {
        "customer_name":    customer["customer_name"],
        "period_days":      days,
        "transactions":     all_transactions,
        "count":            len(all_transactions), #How many transactions you made in a given time period
    }


#FUNCTION DECLERATIONS (what Gemini sees)
# Each FunctionDeclaration is the schema Gemini reads to decide:
#   (a) whether to call this tool at all, and
#   (b) what arguments to fill in.
#
# The "description" fields are the most important part, Gemini reasons about
# them in natural language.The "parameters" block is JSON Schema,
# same structure as OpenAPI.

_GET_BALANCE_DECL = types.FunctionDeclaration(
    name="get_account_balance",
    description=(
        "Returns the current balance for all of the customer's accounts. "
        "Use this when the customer asks about their balance, how much money "
        "they have, or wants an overview of their accounts. "
        "Takes no parameters, the customer is identified automatically from "
        "the authenticated session."
    ),
    parameters={
        "type": "object",
        "properties": {},
    },
)

_GET_TRANSACTION_DECL = types.FunctionDeclaration(
    name="get_recent_transactions",
    description=(
        "Returns recent transactions for the customer across all accounts. "
        "Use this when the customer asks about recent activity, spending, "
        "specific purchases, or transaction history. "
        "The customer is identified automatically from the authenticated "
        "session, do not ask for or include a customer ID."
    ),
    parameters={
        "type": "object",
        "properties": {
            "days": {
                "type":        "integer",
                "description": (
                    "Number of past days to include (default: 30). "
                    "Use ONLY when the customer specifies a time window "
                    "(e.g. 'last 14 days', 'this month'). "
                    "Do NOT use this for count-based queries like 'last 3 transactions'."
                ),
            },
            "limit": {
                "type":        "integer",
                "description": (
                    "Maximum number of transactions to return, by count from most recent. "
                    "Use when the customer asks for 'last N transactions' as a count "
                    "(e.g. 'show me my last 3 transactions' → limit=3, days=30). "
                    "Do NOT set days=3 for this — 'last 3 transactions' means count, not days."
                ),
            },
        },
    },
)

# Package declerations into a Tool object: passed to GenerateContentConfig
ACCOUNT_TOOLS = types.Tool(function_declarations=[_GET_BALANCE_DECL, _GET_TRANSACTION_DECL])

#Dispatch map
# Gemini will returnn a tool name as a string, then map it to the python callable
# function. Any name Gemini retunrs that isnt in the dict is ignored and silent fallback takeover

DISPATCH: dict = {
    "get_account_balance":      get_account_balance,
    "get_recent_transactions":  get_recent_transactions,
}


#ACCOUNT NODE
def account_node(state: EnquiryState) -> dict:
    """
    LangGraph node: answers account enquiries via two-turn function calling.
    Reads from state:  query, customer_id
    Writes to state:   subagent_response, sources, escalated
 
    Two-turn flow:
        Turn 1 → Gemini + tools → FunctionCall (or direct answer)
        Execute → Python function → result dict
        Turn 2 → Gemini + result → natural language answer    
    """
    
    query =         state["query"]
    customer_id =   state["customer_id"]

    print(f"\n[ACCOUNT] Query: '{query[:80]}'")
    print(f"[ACCOUNT] Customer ID: '{customer_id}'")

    try:
        client = get_client()

        #TURN 1: ask Gemini which tool to use
        print("[ACCOUNT] Turn 1: sending user query + tool decleration to Gemini")
        response1 = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[ACCOUNT_TOOLS],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                ), #Prevents gemini from skipping mock_account lookup, this subagent has to go through the db.
                temperature=0,
            ),
        )

        # Gemini will return a string and we need to find the function call part
        # e.g.will be a text part + function_call part. set up code to scall all
        # and take the first function call found.
        function_call = None
        for part in response1.candidates[0].content.parts:
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                function_call = fc
                break
        # If Gemini answered without tool call as query was too vague then return 
        # answer as it is.
        if function_call is None:
            print("[ACCOUNT] Gemini answered directly (no tool call)")
            return {
                "subagent_response":    response1.text.strip(),
                "sources":              [],
                "escalated":            False,
            }
        
        print(f"[ACCOUNT] Gemini requested tool: {function_call.name}({dict(function_call.args)})")
        
        #Execute tool
        tool_args = dict(function_call.args) #dict() converts the proto MapComposite to a plain python dict to unpack the keyword tools

        # Inject customer_id from session state: UNCONDITIONALLY.
        # customer_id is no longer in either function declaration's schema
        # so function_call.args will never contain it. This line is the 
        # single point where identity enters the tool call, and it always 
        # comes from the authenticated session, never from the model's output.
        tool_args["customer_id"] = customer_id

        if function_call.name not in DISPATCH:
            raise ValueError(f"Unknown tool requested: '{function_call.name}'")
        
        tool_result = DISPATCH[function_call.name](**tool_args)
        print(f"[ACCOUNT] tool result: {json.dumps(tool_result)[:120]}...")


        #TURN 2: send tool result back and get natural language with Gemini
        # Reconstruct the full conversation so Gemini has context:
        #   user turn:     the original query
        #   model turn:    Gemini's function call (response1's content)
        #   user turn:     the tool result (FunctionResponse part)
        # Gemini then composes a final answer grounded in the real account data.

        print("[ACCOUNT] Turn 2: sending tool result back to Gemini")
        response2 = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=query)],
                ),
                response1.candidates[0].content, #Model function call
                types.Content(
                    role="user",
                    parts=[types.Part(
                        function_response=types.FunctionResponse(
                            name=function_call.name,
                            response=tool_result, #plain dict
                        )
                    )],
                ),
            ],
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0,
            ),
        )

        final_answer = response2.text.strip()
        print(f"[ACCOUNT] Final answer: '{final_answer[:100]}...")

        return {
            "subagent_response":    final_answer,
            "sources":              [], #not RAG
            "escalated":            False,
        }

    except Exception as e:
        print(f"[ACCOUNT] ERROR: {type(e).__name__}: {e}")
        return {
            "subagent_response": (
                "I'm unable to retrieve your account information right now. "
                "Please call us on 1300 555 100 or try again shortly "
            ),
            "sources":      [],
            "escalated":    False,
            "error":        f"Account subagent failed {e}",
        }
    

#TEST HARNESS
# Stage A: pure Python tool tests (no API key needed):
#   Verifies mock data loads correctly and tool functions return expected shapes.
#
# Stage B: live node tests (GOOGLE_API_KEY required):
#   Runs account_node end-to-end. Watch for:
#   - Turn 1 log: which tool did Gemini choose for each query?
#   - Turn 2 log: is the final answer grounded in the mock data?
#   - Account numbers should appear as last 4 digits only (e.g. "...4521")
#   - Balance should match mock_accounts.json exactly

if __name__ == "__main__":
    from src.state import make_initial_state
 
    # ── Stage A: Tool function tests (no API) ─────────────────────────────────
    print("=" * 60)
    print("STAGE A — Tool function tests (no Gemini API needed)")
    print("=" * 60)
 
    print("\n[1] get_account_balance — known customer")
    result = get_account_balance("C001")
    assert "accounts" in result, "Expected 'accounts' key"
    assert result["customer_name"] == "Alex Johnson"
    assert len(result["accounts"]) == 2
    for acc in result["accounts"]:
        assert len(acc["account_number_last4"]) == 4, "Account number should be masked to 4 digits"
    print(f"  customer_name : {result['customer_name']}")
    for acc in result["accounts"]:
        print(f"  {acc['account_type']:35s}  ${acc['balance']:,.2f} AUD  (last 4: {acc['account_number_last4']})")
    print("  ✓ Balance lookup and masking correct")
 
    print("\n[2] get_recent_transactions — last 30 days")
    result = get_recent_transactions("C001", days=30)
    assert "transactions" in result
    # All returned transactions should be within the last 30 days
    cutoff = date.today() - timedelta(days=30)
    for txn in result["transactions"]:
        assert date.fromisoformat(txn["date"]) >= cutoff, f"Transaction {txn['date']} outside window"
    print(f"  Transactions in last 30 days: {result['count']}")
    for txn in result["transactions"]:
        sign = "+" if txn["type"] == "credit" else ""
        print(f"  {txn['date']}  {txn['description']:40s}  {sign}${abs(txn['amount']):,.2f}")
    print("  ✓ Date filtering correct — no transactions older than 30 days")
 
    print("\n[3] get_account_balance — unknown customer")
    result = get_account_balance("C999")
    assert "error" in result
    print(f"  error: {result['error']}")
    print("  ✓ Unknown customer handled gracefully")
 
    print("\n[4] get_account_balance — no customer ID")
    result = get_account_balance("")
    assert "error" in result
    print(f"  error: {result['error']}")
    print("  ✓ Empty customer ID handled gracefully")
 
    print("\n✓ All Stage A checks passed\n")
 
    # ── Stage B: Live node tests (needs GOOGLE_API_KEY) ───────────────────────
    if not os.environ.get("GOOGLE_CLOUD_PROJECT"):
        print("Stage B skipped: GOOGLE_CLOUD_PROJECT not set.")
        sys.exit(0)
 
    print("=" * 60)
    print("STAGE B — Live node tests (Gemini API)")
    print("=" * 60)
 
    test_cases = [
        {
            "query":       "What is my current account balance?",
            "customer_id": "C001",
            "note":        "Balance query — expect get_account_balance tool call",
        },
        {
            "query":       "Show me my recent transactions from the last 14 days",
            "customer_id": "C001",
            "note":        "Transaction query — expect get_recent_transactions(days=14)",
        },
        {
            "query":       "What did I spend at the supermarket recently?",
            "customer_id": "C001",
            "note":        "Spending query — expect transactions tool, Gemini filters by merchant",
        },
        {
            "query":       "What is my balance?",
            "customer_id": "",
            "note":        "No auth — tool should return error, Gemini should redirect",
        },
    ]
 
    for i, tc in enumerate(test_cases, 1):
        print(f"\n{'─' * 60}")
        print(f"[{i}/{len(test_cases)}] {tc['note']}")
        print(f"Query:       '{tc['query']}'")
        print(f"Customer ID: '{tc['customer_id']}'")
 
        state  = make_initial_state(tc["query"], customer_id=tc["customer_id"])
        result = account_node(state)
 
        print(f"\nsubagent_response:\n{result['subagent_response']}")
        if result.get("error"):
            print(f"error: {result['error']}")
 
    print(f"\n{'=' * 60}")
    print("Stage B complete. Check responses above.")
    print("Account numbers should appear as last 4 digits only.")
    print("Balances should match mock_accounts.json exactly.")
    print("=" * 60)