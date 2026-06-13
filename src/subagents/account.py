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
from google import genai
from google.genai import types

from src.state import EnquiryState

load_dotenv()


#CONFIG
GEMINI_MODEL = "gemini-2.5-flash"
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
   1300 555 100 or visit a branch.\
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

def get_recent_transactions(customer_id: str, days: int= 30) -> dict:
    """
    Returns transactions from the last N days across all of the customer's accounts.
 
    Transactions are sorted by date descending (most recent first).
    The cutoff date is calculated from today's date at call time.
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
        for txn in acc.get("transactions, []"):
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
        "they have, or wants an overview of their accounts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {
                "type":         "string",
                "description":  "The customer's unique ID (e.g. C001)",
            }
        },
        "required": ["customer_id"],
    },
)

_GET_TRANSACTION_DECL = types.FunctionDeclaration(
    name="get_recent_transactions",
    description=(
        "Returns recent transactions for the customer across all accounts, "
        "filtered to the last N days. Use this when the customer asks about "
        "recent activity, spending, specific purchases, or transaction history."
    ),
    parameters={
        "type": "object",
        "properties": {
            "customer_id": {
                "type":         "string",
                "description":  "The customer's unique ID"
            },
            "days": {
                "type":         "integer",
                "description":  "Number of past days to unclude (default: 30).",
            },
        },
        "required": ["customer_id"],
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