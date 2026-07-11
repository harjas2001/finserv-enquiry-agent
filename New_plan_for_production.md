P7a — Vertex AI migration (new step, ~1 evening)
New GCP project, enable APIs, service account, swap the genai.Client calls, rewrite the embeddings wrapper, re-run evals to confirm parity, commit.
P7b — Cloud Run deploy (~1 evening)
Hardened Dockerfile, ChromaDB baked in, push to Artifact Registry, deploy with the service account attached, live URL.
P8 — CI/CD with eval gate (~2 evenings)
GitHub Actions: test → build → deploy to no-traffic revision → run evals against revision → promote on pass. Workload Identity Federation for keyless auth.
P9 — Observability + LLMOps (~2 evenings)
Structured JSON logging, log sink → BigQuery, Looker Studio dashboard, uptime check + alert. Optional nightly eval job if time permits — I'd push for it since it directly hits "model monitoring, continuous improvement" from the JD.