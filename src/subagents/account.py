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