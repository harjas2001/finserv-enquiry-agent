"""
Phase 1 · Step 2 — Swap LLM to Gemini 2.5 Flash
=================================================
Goal: replace OpenAI with Google Gemini and confirm the entire LangGraph
agent — including tool calling — works identically with zero changes to
the graph, state, or routing logic.

This is the point of the LangChain abstraction: the orchestration layer
(LangGraph) is completely decoupled from the LLM provider. The only thing
that changes is the three lines that instantiate the LLM.

What changes vs step1:
  Line 1 — import:  ChatOpenAI → ChatGoogleGenerativeAI
  Line 2 — model:   "gpt-4o-mini" → "gemini-2.5-flash"
  Line 3 — env var: OPENAI_API_KEY → GOOGLE_API_KEY

Everything else — AgentState, @tool, ToolNode, StateGraph, should_continue,
add_conditional_edges — is identical. Read through and confirm this for yourself.

Before running:
  pip install -r requirements.txt    (picks up langchain-google-genai)
  Add GOOGLE_API_KEY to your .env    (get it from aistudio.google.com)
"""

import math
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI   # ← CHANGED (was langchain_openai)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# QUICKSTART SANITY CHECK
# ─────────────────────────────────────────────────────────────────────────────
# Before running the full agent, verify your Gemini API key works with a direct
# call. If this fails, fix the key before debugging the agent.
# Comment this out once confirmed working.

def gemini_quickstart() -> None:
    """Raw Gemini API call — no LangGraph, no tools. Just confirms the key works."""
    print("── Gemini quickstart ──────────────────────────────────────")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    response = llm.invoke("In one sentence, what is LangGraph?")
    print(f"Gemini says: {response.content}")
    print("── Quickstart passed ──────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1. STATE — identical to Step 1
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOOLS — identical to Step 1
# ─────────────────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the numeric result as a string.
    Use this whenever the user asks you to compute, calculate, or work out a number.

    Supports standard arithmetic (+, -, *, /, **) and math functions
    (sqrt, sin, cos, log, etc.).

    Args:
        expression: A valid Python math expression.
                    Examples: '247 * 38', '4800 * 0.15', 'sqrt(144)', '(100 + 50) / 3'

    Returns:
        The result as a string, or an error message if the expression is invalid.
    """
    try:
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_env)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"


tools = [calculator]


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM — THE THREE-LINE SWAP
# ─────────────────────────────────────────────────────────────────────────────
#
# Before (Step 1):
#   from langchain_openai import ChatOpenAI
#   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
#
# After (Step 2):
#   from langchain_google_genai import ChatGoogleGenerativeAI
#   llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
#
# bind_tools() call is identical — LangChain handles the provider-specific
# tool schema formatting internally (OpenAI function calling format vs
# Gemini function declaration format). You never see the difference.
#
# GOOGLE_API_KEY is read automatically from the environment by the SDK.
# Get yours at: https://aistudio.google.com/apikey  (free tier is enough for dev)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)  # ← CHANGED
llm_with_tools = llm.bind_tools(tools)


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT EXTRACTION HELPER
# ─────────────────────────────────────────────────────────────────────────────
# Gemini 2.5 Flash is a thinking model. Its response.content can be either:
#   - A plain string  (simple responses, older models)
#   - A list of dicts (thinking models — content blocks with type/text/extras)
#     e.g. [{'type': 'text', 'text': 'The answer is...', 'extras': {'signature': ...}}]
#
# The 'signature' in extras is Gemini's internal reasoning trace — not user-facing.
# This helper normalises both formats into a plain string so the rest of the
# code doesn't need to care which format came back.
#
# This is a behavioural difference from OpenAI worth noting for Slide 5.

def extract_text(content) -> str:
    """Normalise Gemini content to a plain string regardless of format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content)


# ─────────────────────────────────────────────────────────────────────────────
# 4–7. NODES, ROUTER, GRAPH — identical to Step 1
# ─────────────────────────────────────────────────────────────────────────────

def agent_node(state: AgentState) -> dict:
    print("\n[AGENT] Calling Gemini...")
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"[AGENT] → Tool call: {tc['name']}({tc['args']})")
    else:
        print(f"[AGENT] → Final answer: {extract_text(response.content)[:120]}")
    return {"messages": [response]}


tool_node = ToolNode(tools)


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("[ROUTER] Tool call detected → tools")
        return "tools"
    print("[ROUTER] No tool call → end")
    return "end"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    return graph.compile()


# ─────────────────────────────────────────────────────────────────────────────
# 8. RUN + TRACE
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(user_message: str) -> str:
    print(f"\n{'═' * 60}")
    print(f"  USER: {user_message}")
    print(f"{'═' * 60}")

    graph = build_graph()
    initial_state = {"messages": [HumanMessage(content=user_message)]}

    final_answer = ""
    for step_num, step_update in enumerate(graph.stream(initial_state, stream_mode="updates"), start=1):
        print(f"\n  ── Step {step_num} ──────────────────────────────────────")
        for node_name, node_output in step_update.items():
            print(f"  Node: [{node_name}]")
            for msg in node_output.get("messages", []):
                msg_type = type(msg).__name__
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"    {msg_type}: calling {tc['name']}({tc['args']})")
                elif hasattr(msg, "tool_call_id"):
                    print(f"    {msg_type}: result = {msg.content}")
                elif hasattr(msg, "content") and msg.content:
                    clean = extract_text(msg.content)
                    print(f"    {msg_type}: {clean}")
                    final_answer = clean

    print(f"\n{'─' * 60}")
    print(f"  FINAL: {final_answer}")
    print(f"{'─' * 60}\n")
    return final_answer


# ─────────────────────────────────────────────────────────────────────────────
# BEHAVIOURAL DIFFERENCES TO WATCH FOR
# ─────────────────────────────────────────────────────────────────────────────
# Run the same three tests as Step 1 and compare:
#
# 1. Tool calling — does Gemini call the calculator on the same queries?
#    Gemini 2.5 Flash is generally strong at tool use, but may phrase the
#    tool call args differently (e.g. '247*38' vs '247 * 38').
#
# 2. Response verbosity — Gemini tends to be more conversational than GPT-4o-mini.
#    It may add explanation around the answer. Note this — it's relevant for the
#    guardrail layer in Phase 4 (verbose answers increase token cost per turn).
#
# 3. Multi-step reasoning (Test 3) — does Gemini do one compound tool call or two?
#    Compare with what GPT-4o-mini did. Neither is wrong; they reflect different
#    reasoning strategies. Document this — it becomes Slide 5 prompt engineering evidence.
#
# 4. Direct answer quality (Test 2) — Gemini may elaborate more on "capital of Australia".
#    Same information, different style.
#
# Save screenshots or copy-paste the terminal output for each test.
# These before/after comparisons are your prompt engineering rubric evidence.

if __name__ == "__main__":
    gemini_quickstart()

    # Same three tests as Step 1 — compare the traces
    run_agent("What is 247 multiplied by 38?")
    run_agent("What is the capital of Australia?")
    run_agent(
        "First, what is 15% of 4800? "
        "Then take that result and divide it by 12. "
        "Walk me through each calculation."
    )