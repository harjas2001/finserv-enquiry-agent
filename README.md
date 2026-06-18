# finserv-enquiry-agent

Agentic customer enquiry handler for **Clearwater Bank** (fictional) — a multi-agent system using Google Gemini + LangGraph with RAG, Responsible AI guardrails, and an offline LLM eval harness.

Built as a portfolio project demonstrating production-grade agentic AI patterns: intent routing, tool calling, retrieval-augmented generation, human-in-the-loop escalation, PII redaction, and automated quality gating.

---

## Architecture

One **orchestrator** classifies intent and routes to one of four **specialist subagents**. Every response passes through a **guardrail node** before being returned.

```
User query
    │
    ▼
Orchestrator (gemini-3.5-flash, thinking_budget=0)
    │  classifies intent
    ├─── account_enquiry  → Account Subagent    (tool call → mock_accounts.json)
    ├─── product_info     → Product Subagent    (RAG → ChromaDB → 4 Clearwater PDFs)
    ├─── complaint        → Complaint Subagent  (HITL interrupt → escalation)
    └─── out_of_scope     → Deflector Subagent  (pattern match → safe refusal)
                                    │
                                    ▼
                            Guardrail Node
                         (PII redact · hallucination flag · scope block)
                                    │
                                    ▼
                              FastAPI response
```

**State** flows through a LangGraph `StateGraph` using a custom `EnquiryState` TypedDict (12 fields). Conversation memory is persisted across turns with `MemorySaver`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph` + `MemorySaver` |
| LLM — orchestrator | `gemini-2.5-flash` (`thinking_budget=0`) |
| LLM — subagents | `gemini-2.5-flash-lite` |
| LLM — eval judge | `gemini-2.5-flash` |
| Embeddings | `gemini-embedding-2` (3072d) via custom `GeminiEmbeddings` class |
| Vector store | ChromaDB (local) |
| Guardrails | `src/guardrails.py` — 3 checkers |
| Eval harness | `evals/run_evals.py` — 20 labelled cases, 2 CI gates |
| API | FastAPI |
| Demo UI | Streamlit two-panel app |
| Deployment target | GCP Cloud Run |

---

## Setup

**Prerequisites:** Python 3.11+, a Google AI Studio API key (free tier works).

```bash
git clone https://github.com/<your-handle>/finserv-enquiry-agent.git
cd finserv-enquiry-agent
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_key_here
```

Ingest the knowledge base (one-time):

```bash
python -m src.rag.ingest
```

This chunks the four Clearwater Bank PDFs, generates embeddings via `gemini-embedding-2`, and upserts 15 chunks into a local ChromaDB collection.

---

## Running

**API server:**

```bash
uvicorn api.main:app --reload --port 8000
```

**Streamlit demo UI** (requires the API to be running):

```bash
streamlit run src/ui/app.py
```

The UI has two panels: left is the customer chat window (with a C001 / C002 / Guest selector), right is a live system panel showing intent, subagent, RAG sources, and guardrail flags per response.

**Direct API call:**

```bash
curl -X POST http://localhost:8000/enquire \
  -H "Content-Type: application/json" \
  -d '{"query": "What is my account balance?", "customer_id": "C001", "session_id": "abc123"}'
```

---

## Five Routing Paths

| Query | Customer | Route | What it demonstrates |
|---|---|---|---|
| "What is my current account balance?" | C001 | account → tool call | Tool calling with `mode="ANY"`, masked account numbers |
| "What are your current home loan interest rates?" | C001 | product → RAG | Grounded retrieval, sources returned |
| "What is your exchange rate for converting AUD to USD?" | C001 | product → relevance gate → deflect | Relevance threshold (0.75), graceful empty-source deflection |
| "There is a $450 transaction I never made on my card" | C002 | complaint → HITL | LangGraph `interrupt()`, case logged, escalation banner |
| "Can you give me tips on investing my super?" | C001 | out_of_scope → deflector | Scope guardrail, allow_retry flag |

---

## Key Design Decisions

**LangGraph over alternatives** — chosen for its custom `StateGraph` with TypedDict state schema and conditional edge routing. State channels must be declared in the TypedDict class body; factory defaults alone are insufficient for LangGraph to register fields.

**Model tiering** — orchestrator and eval judge use `gemini-3.5-flash` for reliability. All four subagents use `gemini-2.5-flash-lite` for cost. `thinking_budget=0` on the orchestrator disables chain-of-thought for faster classification. Note: `gemini-2.0-flash` was retired June 2026; `gemini-2.5-pro` is paid-only as of April 2026.

**Tool calling `mode="ANY"`** — required to force Gemini to use a tool. `mode="AUTO"` allows the model to skip tool calls silently, causing the account subagent to hallucinate balances from training data.

**PII: redact in-place, don't refuse** — BSBs are replaced with `[BSB REDACTED]`; account numbers are masked to last 4 digits (`****6789`). Blanket refusal degrades response utility; in-place redaction preserves it while enforcing data minimisation.

**Runtime vs offline separation** — guardrails (`src/guardrails.py`) run synchronously on every request: fast regex and term-overlap checks, no extra API calls. LLM-as-judge faithfulness scoring runs offline in the eval harness, where latency isn't a constraint. These two concerns are deliberately kept separate.

**RAG relevance threshold** — set at 0.75 cosine similarity after empirical tuning. The initial value of 0.80 was too strict: the FX rate query scored 0.7945 and incorrectly passed the gate, causing a downstream faithfulness failure. The product subagent returns an empty-source deflection when no chunk clears the threshold.

---

## Guardrail Layer

Three checks run in sequence on every `subagent_response`:

| Check | Type | Trigger | Action |
|---|---|---|---|
| PII detection | Hard block | BSB pattern or 6–10 digit account number | Redact in-place |
| Hallucination risk | Soft flag | RAG path + key-term overlap < 5% | Set flag, serve response |
| Scope violation | Hard block | Financial advice language regex | Replace with safe fallback |

PII overrides scope when both fire simultaneously — leaking account numbers is a higher-severity compliance event than generating advice language.

---

## Eval Harness

```bash
python -m evals.run_evals
```

Runs 20 labelled test cases through the full LangGraph graph across three phases:

- **Phase 1** — graph execution for all 20 cases (~20 Gemini calls)
- **Phase 2** — faithfulness judge on RAG cases (~5 judge calls)
- **Phase 3** — task completion judge on non-HITL cases (~18 judge calls)

**Final scores:**

| Metric | Score | Gate | Result |
|---|---|---|---|
| routing_accuracy | 100% | ≥ 90% | ✅ PASS |
| faithfulness | 100% | ≥ 85% | ✅ PASS |
| task_completion | 85% | — | Reported |
| content_match | 90% | — | Reported |

The harness exits non-zero if either gate is breached — designed to block deployment in a Cloud Build CI pipeline.

**LLM-as-judge prompt note:** deflection responses require an explicit clause in the judge prompt; without it the judge incorrectly penalises correct out-of-scope handling.

---

## Test Suites

All tests run from the project root:

```bash
python -m pytest tests/ -v
```

| Suite | Cases | Result |
|---|---|---|
| `test_orchestrator.py` | 6 classification cases | 6/6 |
| `test_account.py` (A+B) | 8 tool-call assertions | 8/8 |
| `test_complaint.py` (A+B) | 11 HITL assertions | 11/11 |
| `test_deflector.py` | 6 queries × 7 assertions | 42/42 |
| `test_graph.py` (A+B) | Structural + 5 end-to-end paths | PASS |
| `test_api.py` (A+B) | Structural + 5 HTTP paths | PASS |
| `test_guardrails.py` (A+B) | 13 unit + 5 integration | 18/18 |

Each file follows a Stage A (pure Python, no LLM) + Stage B (live graph) pattern to keep fast unit checks separate from API-hitting integration tests.

---

## GCP Deployment

The API is designed for GCP Cloud Run. A `Dockerfile` at the project root builds the FastAPI app:

```bash
gcloud builds submit --tag gcr.io/<project-id>/finserv-enquiry-agent
gcloud run deploy finserv-enquiry-agent \
  --image gcr.io/<project-id>/finserv-enquiry-agent \
  --platform managed \
  --region australia-southeast1 \
  --set-env-vars GOOGLE_API_KEY=<your-key>
```

`GET /health` serves as the Cloud Run liveness probe.

In a production architecture, the `GOOGLE_API_KEY` would be injected from Secret Manager, and the eval harness would run as a Cloud Build step gating each deployment.

---

## Data

All data is synthetic. No real customer records are used.

`mock_accounts.json` — two customers (`C001`, `C002`), two accounts each (savings + credit). Account numbers and BSBs are fictitious.

`complaints.json` — written at runtime by the complaint subagent. Each logged complaint gets a UUID, timestamp, severity classification, and the customer's raw query.

The four Clearwater Bank PDFs are synthetic product guides generated for this project. Using synthetic data gives full control over ground truth for eval testing — every correct answer for product queries is known in advance.

---

## License

MIT