"""Streamlit UI for the policy-aware support agent.

Thin layer only: calls src.agent.answer() for every question and reads
src.config / src.llm / src.retriever purely to wire up provider selection
and display. No guard/retrieval/verification logic lives here.

Run with: streamlit run app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from src.agent import answer
from src.config import LLMConnection, get_config
from src.llm import LLMClient
from src.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = REPO_ROOT / "data" / "questions.json"

PROVIDER_DEFAULTS = {
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-120b"},
    "Ollama (local)": {"base_url": "http://localhost:11434/v1", "model": "qwen3.5:9b"},
}

st.set_page_config(page_title="IIFL Policy Agent", layout="wide")


@st.cache_resource
def get_retriever() -> Retriever:
    """Load the retriever once per server process — the embedding model
    load is the slow part, so this avoids repeating it on every rerun."""
    config = get_config()
    return Retriever(config.policies_dir, config.retrieval)


get_retriever()  # warm the cache at startup so the first question isn't slow


def _sample_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def _build_llm_client(provider: str, model_name: str) -> LLMClient:
    """Translate the sidebar's provider selection into an LLMClient."""
    defaults = PROVIDER_DEFAULTS[provider]
    api_key = os.environ.get("LLM_API_KEY", "") if provider == "Groq" else "ollama"
    connection = LLMConnection(base_url=defaults["base_url"], api_key=api_key, model=model_name)
    return LLMClient(connection, get_config().llm)


def _load_traces() -> list[dict]:
    logs_path = get_config().logs_path
    if not logs_path.exists():
        return []
    lines = logs_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- Sidebar: provider switch ---
st.sidebar.header("LLM Provider")
current_base_url = os.environ.get("LLM_BASE_URL", PROVIDER_DEFAULTS["Groq"]["base_url"])
provider_names = list(PROVIDER_DEFAULTS.keys())
default_provider = "Ollama (local)" if "11434" in current_base_url else "Groq"
provider = st.sidebar.selectbox("Provider", provider_names, index=provider_names.index(default_provider))
default_model = os.environ.get("LLM_MODEL", PROVIDER_DEFAULTS[provider]["model"])
model_name = st.sidebar.text_input("Model name", value=default_model)
st.sidebar.caption(f"Endpoint: {PROVIDER_DEFAULTS[provider]['base_url']}")

ask_tab, metrics_tab = st.tabs(["Ask", "Metrics"])

with ask_tab:
    st.subheader("Ask a policy question")

    if "question_text" not in st.session_state:
        st.session_state.question_text = ""

    st.write("Sample questions:")
    samples = _sample_questions()
    cols = st.columns(len(samples))
    for col, item in zip(cols, samples):
        label = item["question"] if len(item["question"]) <= 40 else item["question"][:37] + "..."
        if col.button(label, key=item["question"], help=item["question"]):
            st.session_state.question_text = item["question"]

    question = st.text_area("Question", value=st.session_state.question_text, height=80)
    submitted = st.button("Get Answer", type="primary")

    if submitted and question.strip():
        try:
            llm_client = _build_llm_client(provider, model_name)
            with st.spinner("Thinking..."):
                result = answer(question, llm_client=llm_client)
        except Exception as exc:
            st.error(f"Could not reach the LLM provider: {exc}")
        else:
            badge = "🟢 RESPOND" if result.action == "respond" else "🔴 ESCALATE"
            st.markdown(f"### {badge} — confidence: {result.confidence}")
            st.write(result.answer)
            st.json(result.model_dump())

            traces = _load_traces()
            last_trace = traces[-1] if traces else None
            with st.expander("Why this answer"):
                if last_trace is None:
                    st.write("No trace available.")
                else:
                    st.write("**Retrieved sections:**")
                    for r in last_trace["retrieved"]:
                        st.write(f"- `{r['section_id']}` — score {r['score']:.3f}")
                    st.write("**Guard & verify checks:**")
                    for check_name, passed in last_trace["checks"].items():
                        icon = "✅" if passed else "❌"
                        st.write(f"{icon} {check_name}: {passed}")
                    if last_trace.get("reason"):
                        st.write(f"**Escalation reason:** `{last_trace['reason']}`")
    elif submitted:
        st.warning("Please enter a question.")

with metrics_tab:
    st.subheader("Evaluation")
    if st.button("Run eval"):
        from eval.run import _failed_cases, _load_cases, _run_cases, _write_results_md, compute_metrics

        with st.spinner("Running eval cases against the live pipeline..."):
            config = get_config()
            cases = _load_cases()
            results, traces = _run_cases(config)
            metrics = compute_metrics(cases, results, traces, get_retriever())
            failed = _failed_cases(cases, results, traces)
            _write_results_md(metrics, failed)

        st.write("### Metrics")
        st.table(
            {
                "Metric": [
                    "Routing accuracy",
                    "False-respond rate",
                    "Retrieval hit@3",
                    "Category accuracy",
                    "Groundedness rate",
                    "JSON retry rate",
                    "Latency p50 (ms)",
                    "Latency p95 (ms)",
                ],
                "Value": [
                    f"{metrics['routing_accuracy']:.0%}",
                    f"{metrics['false_respond_rate']:.0%} "
                    f"({metrics['false_respond_count']}/{metrics['should_escalate_count']})",
                    f"{metrics['retrieval_hit_at_3']:.0%}",
                    f"{metrics['category_accuracy']:.0%}",
                    f"{metrics['groundedness_rate']:.0%}",
                    f"{metrics['json_retry_rate']:.0%}",
                    f"{metrics['latency_p50_ms']:.0f}",
                    f"{metrics['latency_p95_ms']:.0f}",
                ],
            }
        )

        st.write("### Failed cases")
        if not failed:
            st.success("None — every case routed to the expected action.")
        else:
            for f in failed:
                st.write(
                    f"- **{f['question']}** — expected `{f['expected_action']}`, "
                    f"got `{f['actual_action']}` (reason: `{f['reason']}`)"
                )

    st.divider()
    st.write("### Trace history (logs/traces.jsonl)")
    traces = _load_traces()
    if not traces:
        st.write("No traces yet — ask a question or run the eval.")
    else:
        st.write(f"**Action breakdown** ({len(traces)} total requests)")
        action_counts: dict[str, int] = {}
        for t in traces:
            action_counts[t["action"]] = action_counts.get(t["action"], 0) + 1
        st.bar_chart(action_counts)

        st.write("**Latency (ms) per request, most recent last**")
        st.line_chart([t["total_latency_ms"] for t in traces])
