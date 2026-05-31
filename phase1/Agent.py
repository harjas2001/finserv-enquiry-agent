"""
Phase 1 · Step 3 — Google ADK Quickstart
=========================================
Goal: run one ADK agent locally, understand what ADK gives you vs LangGraph,
and confirm the architectural decision for the project and slides.

ADK (Agent Development Kit) is Google's opinionated framework for building
and deploying agents on Vertex AI. It handles a lot of what you had to wire
manually in LangGraph — web UI, session management, eval CLI, deployment.

The trade-off: less control over graph topology in exchange for more
production tooling out of the box.

This file builds the same calculator agent from Steps 1 and 2 in ADK's style.
Comparing the two implementations is the fastest way to see the difference.

─── How to run ───────────────────────────────────────────────────────────────

    cd finserv-enquiry-agent/phase1      ← IMPORTANT: parent of the agent folder
    adk web                              ← launches browser UI on localhost:8000

Then open http://localhost:8000 in your browser, select step3_adk_quickstart,
and type a message. You get a full chat interface with no extra code.

Alternatively, run in CLI mode:
    adk run step3_adk_quickstart

─── What ADK needs ───────────────────────────────────────────────────────────

Unlike LangGraph where you build the graph yourself, ADK needs:
  1. A directory with __init__.py (marks it as an agent module)
  2. An agent.py with a `root_agent` variable (the entry point ADK looks for)
  3. Tools as plain Python functions with descriptive docstrings
  4. GOOGLE_API_KEY in your environment (same key as Step 2)

That's it. No StateGraph, no nodes, no edges, no routing function.
ADK infers the ReAct loop internally.

─── Key difference in tool definition ───────────────────────────────────────

LangGraph (Steps 1–2): used @tool from langchain_core.tools
ADK (Step 3):          plain Python function — no decorator needed

ADK reads the function signature and docstring directly to build the tool
schema for Gemini. The result is identical; the mechanism is different.
"""

import math

from google.adk.agents import Agent


# ─────────────────────────────────────────────────────────────────────────────
# TOOL — plain Python function, no decorator
# ─────────────────────────────────────────────────────────────────────────────
# ADK passes this function's name, docstring, and type hints directly to Gemini
# as a function declaration. Same end result as bind_tools() in LangGraph,
# just without the @tool decorator.

def calculator(expression: str) -> dict:
    """
    Evaluates a mathematical expression and returns the numeric result.
    Use this whenever the user asks you to compute or calculate something.

    Supports standard arithmetic (+, -, *, /) and math functions
    (sqrt, sin, cos, log, etc.).

    Args:
        expression: A valid Python math expression, e.g. '247 * 38', '4800 * 0.15'

    Returns:
        A dict with a 'result' key containing the computed value as a string,
        or an 'error' key if the expression is invalid.
    """
    try:
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_env)  # noqa: S307
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# ROOT AGENT — the ADK entry point
# ─────────────────────────────────────────────────────────────────────────────
# Agent() replaces everything you built manually in LangGraph:
#   - StateGraph definition
#   - agent_node function
#   - tool_node
#   - should_continue routing function
#   - add_conditional_edges wiring
#   - graph.compile()
#
# ADK handles the ReAct loop internally. You describe what the agent is and
# what tools it has — ADK figures out how to run it.
#
# `instruction` is the system prompt — equivalent to the system message you'd
# prepend to state["messages"] in LangGraph. This is where prompt engineering
# lives in ADK agents.
#
# The variable MUST be named `root_agent` — ADK's discovery mechanism looks
# for this exact name in agent.py.




# ─────────────────────────────────────────────────────────────────────────────
# WHAT ADK GIVES YOU THAT LANGGRAPH DOESN'T
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Web UI (adk web)
#    A browser-based chat interface that runs your agent instantly.
#    No FastAPI, no frontend code. Useful for stakeholder demos.
#    In the project: this is what you'd use to demo to non-technical reviewers.
#
# 2. Eval CLI (adk eval)
#    Built-in evaluation runner — provide test cases as JSON, ADK runs them
#    and scores responses. Less flexible than the custom harness you'll build
#    in Phase 4, but zero setup.
#
# 3. Memory Bank
#    Persistent cross-session memory — the agent remembers previous conversations
#    automatically. In LangGraph you'd build this yourself with a checkpointer.
#
# 4. Native Vertex AI Agent Engine deployment
#    `adk deploy` pushes the agent to Vertex AI Agent Engine with one command.
#    Handles scaling, session management, monitoring. No Cloud Run config needed.
#
# 5. Session management
#    ADK tracks conversation history across turns automatically.
#    In LangGraph you manage state explicitly (the messages list in AgentState).
#
# ─────────────────────────────────────────────────────────────────────────────
# WHAT LANGGRAPH GIVES YOU THAT ADK DOESN'T
# ─────────────────────────────────────────────────────────────────────────────
#
# 1. Fine-grained graph control
#    Custom nodes, edges, conditional routing. You decide exactly what happens
#    at every step. ADK's internal loop is a black box you can't customise.
#
# 2. Custom state schema
#    EnquiryState with intent, customer_id, subagent_response, etc.
#    In ADK, state is managed internally — you can't attach custom fields.
#
# 3. Complex multi-agent topologies
#    LangGraph lets you define arbitrary routing: intent → one of four subagents,
#    with different conditions for each path. ADK's sub_agents feature is simpler
#    and less flexible for this pattern.
#
# 4. Provider-agnostic
#    LangGraph works with OpenAI, Gemini, Anthropic, etc.
#    ADK is Google-only (Gemini / Vertex AI).
#
# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURAL DECISION FOR THIS PROJECT (and the slides)
# ─────────────────────────────────────────────────────────────────────────────
#
# Orchestration logic:   LangGraph
#   Reason: the orchestrator needs custom conditional edges to route between
#   four specialist subagents based on intent. EnquiryState needs to carry
#   customer_id, detected intent, and guardrail flags across nodes.
#   LangGraph's explicit graph control is the right tool for this.
#
# Production deployment: ADK (referenced in slides, not built in the POC)
#   Reason: for the Macquarie interview, demonstrating awareness of ADK's
#   Vertex AI Agent Engine deployment path shows production thinking.
#   Slide 7 (GCP Landing Zone) references ADK as the deployment layer.
#   The POC uses Cloud Run + FastAPI; production vision uses ADK.
#
# One-line answer for the interview:
#   "We used LangGraph for orchestration because we needed fine-grained
#    control over multi-agent routing and custom state management. For the
#    production deployment path, we'd leverage Google ADK's Agent Engine
#    integration on Vertex AI, which handles scaling and session management
#    out of the box."