"""
Phase 5 — Streamlit Demo UI
src/ui/app.py
=============
Two-panel demo interface for the Clearwater Bank agentic enquiry handler.
Calls the FastAPI backend at POST /enquire on localhost:8000.

Left panel  — chat window, customer selector, five demo query buttons
Right panel — live system panel: intent, subagent, sources, guardrail flags

Prerequisites:
    FastAPI server must be running:
        uvicorn api.main:app --reload --port 8000

Run this app:
    streamlit run src/ui/app.py

Architecture note:
    This file contains zero business logic. Every routing, guardrail, and RAG
    decision happens inside the LangGraph backend. The UI is purely a
    presentation layer that shapes the JSON response into readable panels.
    That separation is intentional — the same backend serves this Streamlit
    demo, the curl tests, and (in production) the Cloud Run API.
"""

import uuid

import requests
import streamlit as st


# ── Page config ───────────────────────────────────────────────────────────────
# Must be the first Streamlit call in the file.
# layout="wide" enables the two-column layout; without it Streamlit centres a
# narrow column and the system panel has nowhere to go.
st.set_page_config(
    page_title="Clearwater Bank — Enquiry Handler",
    page_icon="🏦",
    layout="wide",
)


# ── Constants ─────────────────────────────────────────────────────────────────

BACKEND_URL = "http://localhost:8000/enquire"

# Display label → API customer_id.
# "Jordan Lee" is the name in mock_accounts.json (C002).
# The PROJECT_HANDOFF lists "Sarah Lee" — mock_accounts.json is the
# source of truth; use Jordan Lee here.
CUSTOMER_OPTIONS: dict[str, str] = {
    "C001 — Alex Johnson":     "C001",
    "C002 — Jordan Lee":       "C002",
    "Guest (unauthenticated)": "",
}
CUSTOMER_LABELS = list(CUSTOMER_OPTIONS.keys())

# Reverse lookup: customer_id → display label (used by demo button auto-switch).
CUSTOMER_ID_TO_LABEL: dict[str, str] = {v: k for k, v in CUSTOMER_OPTIONS.items()}

# Demo queries — one button per demo script query.
# Tuple: (button label, full query text, customer_id to activate for this query)
# The customer_id here is the correct one from the Phase 5 demo script.
DEMO_QUERIES: list[tuple[str, str, str]] = [
    (
        "💰 Balance",
        "What is my current account balance?",
        "C001",
    ),
    (
        "🏠 Home loan rates",
        "What are your current home loan interest rates?",
        "C001",
    ),
    (
        "💱 FX rate",
        "What is your exchange rate for converting AUD to US dollars?",
        "C001",
    ),
    (
        "🚨 Fraud",
        "There is a $450 transaction I never made. Someone has accessed my account.",
        "C002",
    ),
    (
        "📈 Super tips",
        "Can you give me tips on investing my super?",
        "C002",
    ),
]

# Intent → human-readable subagent name shown in the system panel.
SUBAGENT_LABEL: dict[str, str] = {
    "product":      "Product Info Agent (RAG)",
    "account":      "Account Enquiry Agent (Tool Call)",
    "complaint":    "Complaint Handler (HITL)",
    "out_of_scope": "Out-of-Scope Deflector",
}

# Intent → coloured emoji badge (Streamlit doesn't have a native badge widget
# that works everywhere, so emoji + bold markdown is the reliable approach).
INTENT_BADGE: dict[str, str] = {
    "product":      "🟢 **product**",
    "account":      "🔵 **account**",
    "complaint":    "🟠 **complaint**",
    "out_of_scope": "⚫ **out_of_scope**",
}

# Per-subagent token cost estimates from the handoff performance notes.
# These are shown in the system panel and map directly to the capstone slide.
TOKEN_ESTIMATE: dict[str, str] = {
    "product":      "~1,200 tokens",
    "account":      "~800 tokens",
    "complaint":    "~900 tokens",
    "out_of_scope": "0 tokens (no LLM call)",
}


# ── Session state helpers ─────────────────────────────────────────────────────
# All mutable UI state lives in st.session_state so it survives rerenders.
# Streamlit rerenders the entire script on every interaction — session_state
# is the only memory that persists across those reruns within a browser tab.

def _reset_conversation(new_customer_label: str) -> None:
    """
    Clear chat history, generate a fresh session_id, and set the active customer.

    Called when:
      - The app loads for the first time.
      - The user manually changes the customer selector.
      - A demo button activates a different customer than the current one.

    The session_id maps to a LangGraph MemorySaver thread_id, so resetting it
    starts a completely clean graph execution with no prior checkpoint state.
    """
    st.session_state.messages       = []   # chat history list
    st.session_state.last_meta      = None # JSON from the last /enquire response
    st.session_state.session_id     = str(uuid.uuid4())
    st.session_state.customer_label = new_customer_label


# First-ever load — initialise with C001 (Alex Johnson) as default.
if "messages" not in st.session_state:
    _reset_conversation(CUSTOMER_LABELS[0])


# ── Backend integration ───────────────────────────────────────────────────────

def _call_backend(query: str, customer_id: str) -> dict:
    """
    POST {"query", "customer_id", "session_id"} to /enquire.

    Returns the parsed response dict on success, or a dict with an "_error"
    key on any failure. Using a private "_error" key (rather than "error")
    avoids colliding with any field name the backend might add in future.

    Timeout is 60 s — Gemini 2.5 Flash Lite with RAG occasionally takes ~10 s,
    so 60 s gives comfortable headroom without hanging the UI forever.
    """
    payload = {
        "query":       query,
        "customer_id": customer_id,
        "session_id":  st.session_state.session_id,
    }
    try:
        r = requests.post(BACKEND_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    except requests.exceptions.ConnectionError:
        return {"_error": "Cannot reach backend — is the FastAPI server running on port 8000?"}
    except requests.exceptions.Timeout:
        return {"_error": "Request timed out after 60 s. The Gemini API may be slow — try again."}
    except requests.exceptions.HTTPError as exc:
        return {"_error": f"HTTP {exc.response.status_code} — {exc.response.text[:300]}"}
    except Exception as exc:
        return {"_error": f"Unexpected error: {exc}"}


def _submit_query(query: str, customer_id: str) -> None:
    """
    Append user + assistant message pair to chat history and update last_meta.

    This function is called both from the chat input box and from demo buttons.
    It mutates st.session_state directly; the caller is responsible for calling
    st.rerun() afterward so Streamlit redraws with the new state.
    """
    # Append the user's turn immediately — visible even before the API responds.
    st.session_state.messages.append({"role": "user", "content": query})

    data = _call_backend(query, customer_id)

    if "_error" in data:
        # Surface backend errors as a chat message so the demo doesn't silently break.
        st.session_state.messages.append({
            "role":    "assistant",
            "content": f"⚠️ {data['_error']}",
        })
        st.session_state.last_meta = None

    else:
        st.session_state.messages.append({
            "role":    "assistant",
            "content": data["answer"],
        })
        # Keep session_id in sync — the backend echoes whatever session_id it used
        # (either the one we sent or an auto-generated UUID if we sent empty string).
        st.session_state.session_id = data.get("session_id", st.session_state.session_id)
        st.session_state.last_meta  = data


# ── Layout ────────────────────────────────────────────────────────────────────
# 3:2 column split. The chat window needs more horizontal space for message text;
# the system panel is a compact metadata display.

col_chat, col_sys = st.columns([3, 2], gap="large")


# ══════════════════════════════ LEFT PANEL ════════════════════════════════════
with col_chat:

    st.markdown("## 🏦 Clearwater Bank")
    st.caption("Agentic Customer Enquiry Handler — capstone demo")
    st.divider()

    # ── Customer selector ──────────────────────────────────────────────────
    # The index= parameter is what keeps the selectbox in sync when a demo
    # button auto-switches the customer. Without it, the widget would visually
    # show the old customer even after _reset_conversation() ran.
    current_label = st.session_state.customer_label
    selected_label = st.selectbox(
        "Active customer",
        options=CUSTOMER_LABELS,
        index=CUSTOMER_LABELS.index(current_label),
        key="customer_selector",
    )

    # Manual customer switch — reset the session so conversation history doesn't
    # bleed across customers.
    if selected_label != st.session_state.customer_label:
        _reset_conversation(new_customer_label=selected_label)
        st.rerun()

    active_customer_id = CUSTOMER_OPTIONS[selected_label]

    # ── Demo query buttons ─────────────────────────────────────────────────
    # Five buttons across two rows (3 + 2) — less cramped than a single row of 5.
    # Each button pre-sets the right customer and fires the query in one click,
    # which keeps the 5-minute demo video clean and mistake-free.
    st.markdown("**Demo queries:**")
    row_a = st.columns(3)
    row_b = st.columns(2)
    all_cols = row_a + row_b  # flattened list of 5 column objects

    for i, (btn_label, query_text, target_cid) in enumerate(DEMO_QUERIES):
        with all_cols[i]:
            if st.button(btn_label, use_container_width=True, key=f"demo_{i}"):
                # Auto-switch customer if this demo query belongs to a different one.
                # _reset_conversation clears history so the chat stays coherent.
                if target_cid != active_customer_id:
                    target_label = CUSTOMER_ID_TO_LABEL.get(target_cid, CUSTOMER_LABELS[0])
                    _reset_conversation(new_customer_label=target_label)
                _submit_query(query_text, target_cid)
                st.rerun()

    st.divider()

    # ── Chat history ───────────────────────────────────────────────────────
    # Streamlit's st.chat_message() handles the user/assistant avatar styling.
    # We iterate the full history list so older messages stay visible as the
    # conversation grows.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Allow-retry notice — shown after out-of-scope deflections.
    # The backend sets allow_retry=True when the deflector handled the query.
    # We show this as an info banner below the last message, not inside it,
    # so it's visually distinct from the assistant's response text.
    # if st.session_state.last_meta and st.session_state.last_meta.get("allow_retry"):
    #     st.info(
    #         "💬 That topic is outside what I can help with here. "
    #         "Please try a different question."
    #     )

    # ── Chat input ─────────────────────────────────────────────────────────
    # st.chat_input() renders at the bottom of its column automatically.
    # The walrus operator (:=) means "if user typed something, run this block".
    if user_input := st.chat_input("Ask Clearwater Bank..."):
        _submit_query(user_input, active_customer_id)
        st.rerun()


# ══════════════════════════════ RIGHT PANEL ═══════════════════════════════════
with col_sys:

    st.markdown("## 🔍 System Panel")
    st.caption("Live agent metadata — updates on every response")
    st.divider()

    # Placeholder before first query
    if st.session_state.last_meta is None:
        st.markdown("*Send a message or click a demo query to see the agent's internal state.*")

    else:
        meta   = st.session_state.last_meta
        intent = meta.get("intent", "unknown")
        flags  = meta.get("guardrail_flags", {})

        # ── Intent ────────────────────────────────────────────────────────
        st.markdown("#### Intent Classified")
        st.markdown(INTENT_BADGE.get(intent, f"⚪ **{intent}**"))

        # ── Subagent ──────────────────────────────────────────────────────
        st.markdown("#### Subagent")
        # st.info() gives a light-blue box — visually distinct from plain text
        # and easy to read in a screen recording.
        st.info(SUBAGENT_MAP := SUBAGENT_LABEL.get(intent, "Unknown"))

        # ── HITL escalation banner ─────────────────────────────────────────
        # Only shown when the complaint subagent triggered an urgent escalation.
        # st.error() uses a red background — appropriate severity for a fraud case.
        if meta.get("escalated"):
            st.error("🚨 **Case Escalated** — queued for human review (HITL triggered)")

        st.divider()

        # ── Sources (RAG grounding evidence) ──────────────────────────────
        # This is the visual proof of RAG for the capstone rubric.
        # Empty sources = account tool call, complaint, or deflection path.
        st.markdown("#### 📄 Sources Retrieved")
        sources = meta.get("sources", [])
        if sources:
            for src in sources:
                st.markdown(f"- `{src}`")
        else:
            st.caption("None — this path does not use RAG retrieval.")

        st.divider()

        # ── Guardrail flags ────────────────────────────────────────────────
        # Three rows, one per checker in guardrails.py.
        #
        # IMPORTANT — pii_detected tuple bug:
        # guardrails.py line: flags["pii_detected"] = True,
        # The trailing comma creates a tuple (True,) instead of bool True.
        # bool() wrapping handles both bool True and tuple (True,) correctly
        # so the display is correct regardless of whether that bug is fixed.
        st.markdown("#### 🛡 Guardrail Flags")

        pii_raw   = flags.get("pii_detected",      False)
        hal_raw   = flags.get("hallucination_risk", False)
        scope_raw = flags.get("out_of_scope",       False)

        # Defensive cast: bool((True,)) == True, bool(True) == True, bool(False) == False
        pii   = bool(pii_raw)
        hal   = bool(hal_raw)
        scope = bool(scope_raw)

        def _flag_row(label: str, triggered: bool, warn_msg: str, ok_msg: str) -> None:
            """Render one guardrail flag as a two-column label + status row."""
            label_col, status_col = st.columns([1, 2])
            label_col.markdown(f"**{label}**")
            if triggered:
                status_col.warning(f"⚠️ {warn_msg}")
            else:
                status_col.success(f"✅ {ok_msg}")

        _flag_row("PII",           pii,   "Redacted in-place",   "Clear")
        _flag_row("Hallucination", hal,   "Risk flagged (soft)",  "Clean")
        _flag_row("Scope",         scope, "Blocked",              "Clear")

        st.caption(
            "PII: hard block (redact). "
            "Hallucination: soft flag (response served). "
            "Scope: hard block (replaced)."
        )

        st.divider()

        # ── Token estimate ────────────────────────────────────────────────
        # Pre-baked estimates from the handoff performance notes.
        # These map directly to the Performance slide in the capstone deck.
        # Not a live measurement — the backend doesn't instrument token counts
        # in this POC.
        st.markdown("#### ⚡ Token Estimate")
        est = TOKEN_ESTIMATE.get(intent, "N/A")
        st.markdown(f"**Subagent:** {est}")
        st.caption("Orchestrator adds ~500 tokens on every path.")