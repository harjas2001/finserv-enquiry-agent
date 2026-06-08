"""
Phase 3 — Orchestrator Node
src/orchestrator.py
====================
The orchestrator is the first node every enquiry passes through. It does one
thing: classify the customer's query into one of four intent categories, write
that intent to state, and let LangGraph's conditional edge route to the correct
subagent.
 
Graph position:
    START → [orchestrator_node] → route_to_subagent() → subagent node
 
Intent categories and their destinations:
    "account"      → account_node     (tool call → mock account API)
    "product"      → product_node     (RAG pipeline from Phase 2)
    "complaint"    → complaint_node   (tool call + HITL escalation)
    "out_of_scope" → deflector_node   (polite refusal, minimal LLM)
 
This file exports two objects used by graph.py:
    orchestrator_node   — the LangGraph node function
    route_to_subagent   — the conditional edge routing function
 
Phase 1 connection:
    orchestrator_node is structurally identical to agent_node from Phase 1 —
    it reads state, calls Gemini, returns a partial dict. The difference is
    what it does with the response: instead of deciding whether to call a tool,
    it extracts an intent string that determines which node runs next.
 
    route_to_subagent is the direct equivalent of should_continue() from Phase 1.
    Where should_continue returned "tools" or "end", route_to_subagent returns
    one of four subagent keys. Same mechanism, four destinations instead of two.
"""

import os
import sys 
from dotenv import load_dotenv

from google import genai
from google.genai import types
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.state import EnquiryState

load_dotenv()


# CONFIG
GEMINI_MODEL = "gemini-2.5-flash"
VALID_INTENTS = {"account", "product", "complaint", "out_of_scope"}


#CLASSIFICATION PROMPT
SYSTEM_PROMPT = """\
You are an intent classifier for Clearwater Bank's automated customer service system.
 
Classify the customer query into exactly one of these four categories:
 
  account      — Questions about the customer's own accounts: balances, transactions,
                 statements, account history, recent payments, transfer status
 
  product      — Questions about Clearwater Bank's financial products and features:
                 loan interest rates, eligibility criteria, fee schedules, savings
                 account terms, how a product works in general
 
  complaint    — Complaints, disputes, service failures, requests to escalate to a
                 human agent, expressions of dissatisfaction with service or charges
 
  out_of_scope — Anything else: general financial knowledge questions, competitor
                 comparisons, requests for personalised investment advice, topics
                 unrelated to Clearwater Bank products or the customer's own account
 
Rules:
  - If the query fits multiple categories, choose the most specific one.
    Example: "Why has my savings bonus rate disappeared?" → complaint (not product),
    because the customer is expressing dissatisfaction, not asking how the rate works.
  - If the query is vague, ambiguous, or does not fit clearly, choose out_of_scope.
  - Reply with ONLY the category name — no punctuation, no explanation, no other text.\
"""