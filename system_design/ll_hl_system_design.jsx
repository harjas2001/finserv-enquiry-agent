import { useState } from "react";

// ── Theme ─────────────────────────────────────────────────────────────────────
const T = {
  bg: "#050c18", surface: "#0b1526", card: "#0d1b2e",
  border: "#1a2e48", text: "#dce6f0", muted: "#4a6080", dim: "#1e3048",
  user: "#38BDF8", api: "#818CF8", graph: "#0EA5E9", orch: "#22D3EE",
  account: "#34D399", product: "#FCD34D", complaint: "#F87171",
  deflect: "#A78BFA", guardrail: "#FB923C", response: "#4ADE80",
  gcp: "#60A5FA", rag: "#FDE68A", evals: "#C084FC", cicd: "#7DD3FC",
};

const mono = "'IBM Plex Mono', monospace";
const sans = "'DM Sans', sans-serif";

// ── Primitives ─────────────────────────────────────────────────────────────────
const Arr = ({ label, color = T.dim }) => (
  <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "2px 0" }}>
    {label && <div style={{ fontSize: 9, color: T.muted, fontFamily: mono, marginBottom: 2, textAlign: "center" }}>{label}</div>}
    <div style={{ width: 1, height: 20, background: color, opacity: 0.35 }} />
    <div style={{ width: 0, height: 0, borderLeft: "4px solid transparent", borderRight: "4px solid transparent", borderTop: `6px solid ${color}`, opacity: 0.35 }} />
  </div>
);

const Bx = ({ t, s, d, tags, color, sm = false }) => (
  <div style={{
    background: T.card, border: `1px solid ${color}20`, borderLeft: `3px solid ${color}`,
    borderRadius: 5, padding: sm ? "7px 9px" : "10px 13px",
    boxShadow: `0 0 20px ${color}08`, width: "100%",
  }}>
    <div style={{ color, fontSize: sm ? 9 : 10, fontWeight: 700, fontFamily: mono, textTransform: "uppercase", letterSpacing: "0.07em" }}>{t}</div>
    {s && <div style={{ color: T.text, fontSize: sm ? 11 : 12, marginTop: 2, fontFamily: sans }}>{s}</div>}
    {d && <div style={{ color: T.muted, fontSize: 9, marginTop: 3, fontFamily: mono, lineHeight: 1.65, whiteSpace: "pre" }}>{d}</div>}
    {tags && (
      <div style={{ display: "flex", flexWrap: "wrap", gap: 3, marginTop: 5 }}>
        {tags.map((tg, i) => (
          <span key={i} style={{ fontSize: 9, padding: "1px 5px", borderRadius: 2, border: `1px solid ${color}35`, color, background: `${color}10`, fontFamily: mono }}>{tg}</span>
        ))}
      </div>
    )}
  </div>
);

const Dashed = ({ children, color, label }) => (
  <div style={{ border: `1px dashed ${color}30`, borderRadius: 8, padding: "11px 12px", background: `${color}05`, width: "100%" }}>
    {label && <div style={{ fontSize: 9, color, fontFamily: mono, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8, opacity: 0.55 }}>{label}</div>}
    {children}
  </div>
);

const Sec = ({ n, text, color = T.muted }) => (
  <div style={{ fontSize: 9, fontFamily: mono, color, textTransform: "uppercase", letterSpacing: "0.12em", marginBottom: 6, marginTop: 18, opacity: 0.7 }}>
    {n && <span style={{ opacity: 0.4, marginRight: 6 }}>·{n}·</span>}{text}
  </div>
);

const Row2 = ({ children }) => (
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 5 }}>{children}</div>
);

const Row3 = ({ children }) => (
  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 5 }}>{children}</div>
);

// ── Legend dot ─────────────────────────────────────────────────────────────────
const Dot = ({ color, label }) => (
  <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
    <div style={{ width: 8, height: 8, borderRadius: 1, borderLeft: `3px solid ${color}`, background: `${color}20` }} />
    <span style={{ fontSize: 9, color: T.muted, fontFamily: mono }}>{label}</span>
  </div>
);

// ═══════════════════════════════════════════════════════════════════════════════
// SIMPLE VIEW
// ═══════════════════════════════════════════════════════════════════════════════
function Simple() {
  return (
    <div style={{ maxWidth: 500, margin: "0 auto", display: "flex", flexDirection: "column", alignItems: "center" }}>

      <Bx t="Customer" s="User message" color={T.user} />
      <Arr label="HTTP POST /enquire" color={T.api} />
      <Bx t="FastAPI" s="POST /enquire endpoint" d={"GCP Cloud Run · Python"} color={T.api} />
      <Arr color={T.graph} />

      <Dashed color={T.graph} label="LangGraph StateGraph — EnquiryState">
        <div style={{ display: "flex", justifyContent: "center", marginBottom: 10 }}>
          <div style={{ width: "80%" }}>
            <Bx t="Orchestrator" s="Intent Classifier" d={"Gemini 2.5 Flash · temperature=0"} color={T.orch} />
          </div>
        </div>
        <div style={{ fontSize: 9, color: T.muted, fontFamily: mono, letterSpacing: "0.06em", textAlign: "center", marginBottom: 7 }}>
          add_conditional_edges() · routes by detected intent ↓
        </div>
        <Row2>
          <Bx t="Account Enquiry" d={"Tool call\nmock account API"} color={T.account} sm />
          <Bx t="Product Info" d={"RAG pipeline\nChromaDB top-3"} color={T.product} sm />
          <Bx t="Complaint Handler" d={"Tool call\nHITL escalation"} color={T.complaint} sm />
          <Bx t="Out-of-Scope" d={"Scope check\npolite deflect"} color={T.deflect} sm />
        </Row2>
      </Dashed>

      <Arr label="all subagent outputs" color={T.guardrail} />
      <Bx t="Guardrail Layer" s="PII · Hallucination · Scope check" d={"runs on every output before delivery"} color={T.guardrail} />
      <Arr color={T.response} />
      <Bx t="Response" s="Validated customer answer" tags={["{ answer, sources, intent }"]} color={T.response} />

      {/* Legend */}
      <div style={{ marginTop: 28, width: "100%", border: `1px solid ${T.border}`, borderRadius: 6, padding: "10px 12px" }}>
        <div style={{ fontSize: 9, fontFamily: mono, color: T.muted, textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: 8, opacity: 0.6 }}>Legend</div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "5px 14px" }}>
          <Dot color={T.user} label="Customer / I/O" />
          <Dot color={T.api} label="FastAPI (Cloud Run)" />
          <Dot color={T.orch} label="Orchestrator (LangGraph)" />
          <Dot color={T.account} label="Account subagent" />
          <Dot color={T.product} label="Product RAG subagent" />
          <Dot color={T.complaint} label="Complaint + HITL" />
          <Dot color={T.deflect} label="Out-of-scope deflector" />
          <Dot color={T.guardrail} label="Guardrail layer" />
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// DETAILED VIEW
// ═══════════════════════════════════════════════════════════════════════════════
function Detailed() {
  return (
    <div style={{ maxWidth: 560, margin: "0 auto" }}>

      <Sec n="01" text="Input" color={T.user} />
      <Bx
        t="Customer"
        s="User message"
        d={"inbound: account / product / complaint / other"}
        tags={["text input", "session_id"]}
        color={T.user}
      />

      <Arr label="HTTPS POST /enquire · JSON body" color={T.api} />

      <Sec n="02" text="API Layer" color={T.api} />
      <Bx
        t="FastAPI"
        s="POST /enquire endpoint"
        d={"req:  { query: str, session_id: str }\nresp: { answer: str, sources: list, intent: str }"}
        tags={["Cloud Run", "Python 3.12", "uvicorn", "stateless"]}
        color={T.api}
      />

      <Arr color={T.graph} />

      <Sec n="03" text="Orchestration — LangGraph StateGraph" color={T.graph} />
      <Dashed color={T.graph} label="LangGraph StateGraph — compiled graph">

        <Bx
          t="EnquiryState (TypedDict)"
          d={"query:             str\nintent:            str          # routed subagent\ncustomer_id:       str\nsubagent_response: str\nsources:           list[str]\nguardrail_flags:   dict"}
          color={T.graph}
        />

        <Arr color={T.orch} />

        <Bx
          t="Orchestrator Node"
          s="Intent classifier → conditional routing"
          d={"intent classes: account · product · complaint · out-of-scope"}
          tags={["Gemini 2.5 Flash", "temperature=0", "add_conditional_edges()"]}
          color={T.orch}
        />

        <div style={{ fontSize: 9, color: T.muted, fontFamily: mono, textAlign: "center", margin: "8px 0 7px", letterSpacing: "0.06em" }}>
          4 conditional routes ↓
        </div>

        <Row2>
          <Bx
            t="Account Enquiry"
            s="Tool call subagent"
            d={"get_balance(customer_id)\nget_transactions(id, days)\nget_statement(account_id)\n→ mock_accounts.json"}
            tags={["ToolNode", "JSON mock"]}
            color={T.account}
          />
          <Bx
            t="Product Info"
            s="RAG subagent"
            d={"query → gemini-embedding-2\n→ ChromaDB similarity_search\n   k=3, relevance gate >0.80\ngrounding prompt → Gemini"}
            tags={["768-dim", "ChromaDB", "RAG"]}
            color={T.product}
          />
          <Bx
            t="Complaint Handler"
            s="Tool + escalation node"
            d={"log_complaint(id, desc) tool\nLangGraph .interrupt()\nhuman-in-the-loop review\n→ case created / escalated"}
            tags={["HITL", ".interrupt()", "escalation"]}
            color={T.complaint}
          />
          <Bx
            t="Out-of-Scope Deflector"
            s="Scope guardrail"
            d={"intent: off-topic / unsafe?\nclassifier threshold check\npolite deflection response\n— no Gemini call on refuse"}
            tags={["scope guard", "no LLM"]}
            color={T.deflect}
          />
        </Row2>
      </Dashed>

      <Arr label="all subagent outputs →" color={T.guardrail} />

      <Sec n="04" text="Responsible AI — Guardrail Layer" color={T.guardrail} />
      <Dashed color={T.guardrail} label="runs on every output before delivery">
        <Row3>
          <Bx t="PII Detector" d={"regex +\nclassifier\n—\naccount #s\nnames, DOBs\nblocked before\nlogging"} color={T.guardrail} sm />
          <Bx t="Hallucination Check" d={"response\ncites retrieved\ncontext?\n—\nRAG-grounded\ncheck only"} color={T.guardrail} sm />
          <Bx t="Scope Check" d={"refuses\nfinancial\nadvice &\noff-topic\nqueries\n—\nhard block"} color={T.guardrail} sm />
        </Row3>
      </Dashed>

      <Arr color={T.response} />
      <Bx t="Response" s="Validated, PII-clean answer returned to customer" tags={["{ answer, sources, intent }"]} color={T.response} />

      <Sec n="05" text="GCP Infrastructure" color={T.gcp} />
      <Dashed color={T.gcp} label="australia-southeast1 · all services pinned to region">
        <Row2>
          <Bx t="Cloud Run" d={"FastAPI agent API\nautoscale · stateless\nper-request container"} color={T.gcp} sm />
          <Bx t="Vertex AI" d={"Gemini 2.5 Flash\ninference endpoint\n+ Vector Search (prod)"} color={T.gcp} sm />
          <Bx t="BigQuery" d={"conversation logs\neval metric tables\ncost tracking"} color={T.gcp} sm />
          <Bx t="Secret Manager" d={"GOOGLE_API_KEY\nDB credentials\nno secrets in code"} color={T.gcp} sm />
          <Bx t="Cloud Build" d={"CI/CD pipeline\neval gate step\nblocks on failure"} color={T.cicd} sm />
          <Bx t="ChromaDB → Vector Search" d={"local dev: ChromaDB\nprod vision:\nVertex AI Vector Search"} color={T.rag} sm />
        </Row2>
      </Dashed>

      <Sec n="06" text="Evals Harness — Offline · LLMOps" color={T.evals} />
      <Dashed color={T.evals} label="CI/CD eval gate · blocks bad deployments · DeepEval">
        <Bx
          t="Test Dataset"
          d={"20 labelled cases:\n(input, expected_routing, expected_answer)"}
          color={T.evals}
        />
        <div style={{ height: 6 }} />
        <Row3>
          <Bx t="Faithfulness" d={"LLM-as-judge\n—\nGemini Pro\noffline scorer"} color={T.evals} sm />
          <Bx t="Routing Accuracy" d={"deterministic\n—\nexpected vs\nactual intent"} color={T.evals} sm />
          <Bx t="Task Completion" d={"LLM-as-judge\n—\nDeepEval\nscorer"} color={T.evals} sm />
        </Row3>
        <div style={{ height: 6 }} />
        <Bx
          t="Eval Gate"
          s="exit non-zero if scores below threshold → deployment blocked"
          tags={["Cloud Build step", "faithfulness >0.85", "routing >0.90"]}
          color={T.cicd}
        />
      </Dashed>

    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ROOT
// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [tab, setTab] = useState("simple");

  return (
    <div style={{ background: T.bg, minHeight: "100vh", color: T.text, fontFamily: sans }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;500;600&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 3px; }
        ::-webkit-scrollbar-thumb { background: ${T.dim}; border-radius: 2px; }
      `}</style>

      {/* Header */}
      <div style={{ padding: "18px 16px 0", borderBottom: `1px solid ${T.border}` }}>
        <div style={{ fontSize: 9, fontFamily: mono, color: T.muted, letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 3 }}>
          finserv-enquiry-agent · system architecture
        </div>
        <div style={{ fontSize: 17, fontWeight: 600, color: T.text, marginBottom: 8 }}>
          Agentic Customer Enquiry Handler
        </div>
        <div style={{ display: "flex", gap: 4, marginBottom: 12, flexWrap: "wrap" }}>
          {["LangGraph", "Gemini 2.5 Flash", "RAG + ChromaDB", "GCP Cloud Run", "Responsible AI", "LLMOps Evals"].map(b => (
            <span key={b} style={{ fontSize: 8, padding: "2px 6px", borderRadius: 2, border: `1px solid ${T.border}`, color: T.muted, fontFamily: mono }}>
              {b}
            </span>
          ))}
        </div>
        <div style={{ display: "flex" }}>
          {[["simple", "Flow overview"], ["detailed", "Full architecture"]].map(([key, label]) => (
            <button key={key} onClick={() => setTab(key)} style={{
              padding: "7px 16px", background: "transparent", border: "none",
              borderBottom: tab === key ? `2px solid ${T.orch}` : "2px solid transparent",
              color: tab === key ? T.orch : T.muted,
              fontSize: 10, fontFamily: mono, textTransform: "uppercase",
              letterSpacing: "0.06em", cursor: "pointer", fontWeight: tab === key ? 600 : 400,
            }}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: "20px 14px 48px", overflowX: "hidden" }}>
        {tab === "simple" ? <Simple /> : <Detailed />}
      </div>
    </div>
  );
}