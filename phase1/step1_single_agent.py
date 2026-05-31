"""
Phase 1 · Step 1 — LangGraph Single ReAct Agent
================================================
Goal: build the core building blocks of a LangGraph agent from scratch, understand
what each one does, then trace the Thought → Action → Observation loop live.

This file is self-contained — no prior code needed.

The ReAct pattern (Reasoning + Acting):
  1. Agent (think) — LLM receives messages, decides what to do
  2. If it needs information → calls a tool (tool_calls in its response)
  3. Tool executes → result appended to state as a ToolMessage
  4. Back to Agent with the new context → reasons again
  5. When Agent has enough → returns a final answer, loop ends

This loop is the DNA of every subagent you'll build in Phase 3.
"""

import math
import os
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# 1. AGENT STATE
# ─────────────────────────────────────────────────────────────────────────────
#
# State is the single source of truth that every node reads from and writes to.
# Think of it as the "working memory" of the graph — it travels through every
# step and accumulates everything the agent knows so far.
#
# Why TypedDict?
#   LangGraph needs typed state so it knows how to merge updates from each node.
#   Plain dicts would work structurally, but TypedDict gives you autocompletion
#   and makes intent explicit — essential when the state schema grows in Phase 3.
#
# Why Annotated[list, add_messages]?
#   The second argument to Annotated is a "reducer" — it tells LangGraph what to
#   do when a node returns an update to this field.
#   - Without a reducer: the new value replaces the old one entirely.
#   - add_messages: appends new messages to the existing list.
#   This is critical — you want each agent/tool response added to history,
#   not the history wiped on every step.
#
# Connection to Phase 3:
#   EnquiryState will extend this with fields like:
#     intent: str          → which subagent to route to
#     customer_id: str     → resolved from the session
#     subagent_response    → the specialist's answer before guardrail check


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────────────────────────
# 2. TOOLS
# ─────────────────────────────────────────────────────────────────────────────
#
# Tools are Python functions the LLM can decide to call at runtime.
# The LLM never executes tools directly — it generates a tool_call specification
# (name + args) in its response, and LangGraph's ToolNode actually runs it.
#
# The @tool decorator does three things:
#   1. Registers the function name as the tool's identifier
#   2. Parses the docstring into a description the LLM reads when deciding
#      whether to use this tool
#   3. Infers the input schema from type hints so the LLM knows what args to pass
#
# Practical mapping to the project:
#   - Phase 1: calculator (demonstrates the pattern)
#   - Phase 3 Account subagent: get_account_balance(customer_id: str) → same pattern,
#     just hits a mock JSON file instead of doing math
#   - Phase 3 Complaint subagent: log_complaint(customer_id, description) → same pattern


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
        # Restrict execution context to math functions only.
        # __builtins__: {} removes access to open(), exec(), import, etc.
        # Passing math.__dict__ gives access to sqrt, sin, cos, log, pi, etc.
        safe_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, safe_env)  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Calculator error: {e}"


# All tools the agent can use — passed to both bind_tools() and ToolNode()
tools = [calculator]


# ─────────────────────────────────────────────────────────────────────────────
# 3. LLM WITH TOOLS BOUND
# ─────────────────────────────────────────────────────────────────────────────
#
# bind_tools() injects the tool schemas (name, description, input JSON schema)
# into the LLM's context so it knows the tools exist and how to call them.
# Without this, the LLM would never know it has a calculator available.
#
# temperature=0: deterministic output — important for routing and tool decisions.
# Stochastic routing in a multi-agent system creates unpredictable behaviour.
#
# In Step 2 this line becomes:
#   from langchain_google_genai import ChatGoogleGenerativeAI
#   llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
# bind_tools() call stays identical — that's the LangChain abstraction working.

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# ─────────────────────────────────────────────────────────────────────────────
# 4. AGENT NODE — the "think" step
# ─────────────────────────────────────────────────────────────────────────────
#
# A node in LangGraph is just a function that:
#   - Takes the current state as its argument
#   - Returns a dict containing only the fields it wants to update
#
# LangGraph merges the return dict into the existing state using the reducers.
# You never return the full state — only the delta.
#
# What the LLM returns:
#   - An AIMessage with content only → final answer, no more tool calls needed
#   - An AIMessage with tool_calls populated → LLM wants to call a tool
#     Example tool_calls: [{'name': 'calculator', 'args': {'expression': '247 * 38'}}]
#
# The agent node doesn't decide what to do next — that's the router's job.
# It just calls the LLM and returns the response.


def agent_node(state: AgentState) -> dict:
    """
    The reasoning node. Calls the LLM with the full message history.
    Returns the LLM's response appended to the message list.
    """
    print("\n[AGENT] Calling LLM with message history...")
    response = llm_with_tools.invoke(state["messages"])

    # Trace output so you can see exactly what the LLM decided
    if response.tool_calls:
        for tc in response.tool_calls:
            print(f"[AGENT] → Tool call: {tc['name']}({tc['args']})")
    else:
        print(f"[AGENT] → Final answer: {response.content[:120]}")

    return {"messages": [response]}


# ─────────────────────────────────────────────────────────────────────────────
# 5. TOOL NODE — the "act" step
# ─────────────────────────────────────────────────────────────────────────────
#
# ToolNode is a LangGraph prebuilt that:
#   1. Reads the last message from state (the AIMessage with tool_calls)
#   2. Finds the matching tool function by name
#   3. Executes it with the LLM-provided args
#   4. Wraps the result in a ToolMessage (which gets added to state via add_messages)
#
# The ToolMessage format:
#   ToolMessage(content="9386", tool_call_id="call_abc123", name="calculator")
# The tool_call_id links back to the original tool_call in the AIMessage —
# this is required by OpenAI and Gemini APIs for multi-tool tracking.
#
# You could implement this manually (iterate tool_calls, call functions, build ToolMessages),
# but ToolNode handles edge cases like parallel tool calls and error wrapping.

tool_node = ToolNode(tools)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ROUTING FUNCTION — the conditional edge
# ─────────────────────────────────────────────────────────────────────────────
#
# After every agent_node execution, the graph needs to decide: loop or stop?
# This function inspects the last message in state and returns a string that
# maps to the next node (or END).
#
# Literal["tools", "end"] is just a type hint — LangGraph uses the actual
# returned string at runtime to look up the destination in the routing map
# you define in add_conditional_edges().
#
# Phase 3 connection:
# The orchestrator uses a nearly identical pattern, but instead of "tools"/"end"
# it returns "account" / "product" / "complaint" / "deflect" based on intent.
# Same mechanism, different destinations.


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Inspects the last message and decides the next step.

    Returns:
        "tools" → the LLM wants to call a tool, go to tool_node
        "end"   → the LLM produced a final answer, terminate the graph
    """
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print("[ROUTER] Tool call detected → tools")
        return "tools"
    print("[ROUTER] No tool call → end")
    return "end"


# ─────────────────────────────────────────────────────────────────────────────
# 7. GRAPH ASSEMBLY
# ─────────────────────────────────────────────────────────────────────────────
#
# StateGraph is the container. You:
#   1. Register nodes (name → function)
#   2. Define the entry point (first node to run)
#   3. Add edges (static: always go here) and conditional edges (dynamic: go here based on state)
#   4. Compile — converts the definition into a runnable Pregel graph
#
# The graph structure for this agent:
#
#   ┌──────────┐
#   │  START   │
#   └────┬─────┘
#        │
#        ▼
#   ┌──────────┐   tool_calls?   ┌───────────┐
#   │  agent   │ ─── yes ──────► │   tools   │
#   └──────────┘                 └─────┬─────┘
#        │                             │
#        │ no tool_calls               │ (always loops back)
#        ▼                             │
#   ┌──────────┐                       │
#   │   END    │ ◄─────────────────────┘
#   └──────────┘
#        ▲
#        │ (agent returns final answer on second+ pass)
#
# This loop is what makes it "agentic" — it keeps reasoning until it's done.


def build_graph():
    """Assembles, wires, and compiles the ReAct graph."""
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Entry point — where execution starts
    graph.set_entry_point("agent")

    # Conditional edge from agent:
    #   call should_continue(state) → get "tools" or "end"
    #   map the result to the next destination node
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",   # → run tool_node
            "end": END,         # → terminate
        },
    )

    # After tool execution, always return to agent
    # (agent needs to reason about the tool result before deciding what to do next)
    graph.add_edge("tools", "agent")

    compiled = graph.compile()
    print("[GRAPH] Compiled successfully.\n")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# 8. RUN + TRACE
# ─────────────────────────────────────────────────────────────────────────────
#
# graph.stream() executes the graph and yields the state delta after each node.
# stream_mode="updates" → yields only what changed (the node's return value),
# not the full accumulated state. Good for tracing step by step.
#
# graph.invoke() would just give you the final state — useful in production,
# but stream() lets you see every intermediate step during development.


def run_agent(user_message: str) -> str:
    """
    Runs the agent with a single user message. Streams and prints each step.
    Returns the final answer as a string.
    """
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
                # Tool call (AIMessage with tool_calls)
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        print(f"    {msg_type}: calling {tc['name']}({tc['args']})")
                # Tool result (ToolMessage)
                elif hasattr(msg, "tool_call_id"):
                    print(f"    {msg_type}: result = {msg.content}")
                # Final answer or human message (AIMessage / HumanMessage)
                elif hasattr(msg, "content") and msg.content:
                    print(f"    {msg_type}: {msg.content}")
                    final_answer = msg.content

    print(f"\n{'─' * 60}")
    print(f"  FINAL: {final_answer}")
    print(f"{'─' * 60}\n")
    return final_answer


# ─────────────────────────────────────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────────────────────────────────────
#
# Three scenarios to trace:
#
# Test 1 — single tool call
#   Expect: agent → tool call → tool result → agent → final answer
#   The LLM cannot do 247×38 reliably in-weights — it should use the calculator.
#
# Test 2 — no tool needed
#   Expect: agent → final answer (single step, no tool_node visit)
#   Factual knowledge doesn't require a tool call.
#
# Test 3 — multi-step tool use
#   Expect: agent → tool call 1 → result → agent → tool call 2 → result → agent → final
#   The LLM needs two calculations and must reason between them.
#
# Before running, check the printed step trace:
#   - How many steps did each test take?
#   - Did the router correctly identify tool vs no-tool cases?
#   - What did each ToolMessage contain?

if __name__ == "__main__":
    # Requires: OPENAI_API_KEY in .env
    run_agent("What is 247 multiplied by 38?")
    run_agent("What is the capital of Australia?")
    run_agent(
        "First, what is 15% of 4800? "
        "Then take that result and divide it by 12. "
        "Walk me through each calculation."
    )