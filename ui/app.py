"""
AI Invoice Auditor - Streamlit Dashboard
Two-page application:
  Page 1: Invoice Dashboard - list, status badges, HTML report viewer, HITL buttons
  Page 2: Invoice Q&A - RAG chat with Triad quality scores

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

# -- Path setup ----------------------------------------------------------------
_root = Path(__file__).parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from dotenv import load_dotenv

load_dotenv(_root / ".env")

# -- Config --------------------------------------------------------------------
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "./outputs/reports"))
INCOMING_DIR = Path(os.getenv("INCOMING_DIR", "./data/incoming"))
DECISIONS_LOG = Path(os.getenv("DECISIONS_LOG", "./logs/human_decisions.json"))
DECISIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# -- Helpers -------------------------------------------------------------------

STATUS_EMOJI = {
    "AUTO_APPROVED": "OK",
    "MANUAL_REVIEW": "WARN",
    "REJECTED": "NO",
}
STATUS_COLOR = {
    "AUTO_APPROVED": "green",
    "MANUAL_REVIEW": "orange",
    "REJECTED": "red",
}


def _load_reports() -> list[dict]:
    """Scan REPORTS_DIR and parse basic metadata from report filenames."""
    reports = []
    for f in sorted(REPORTS_DIR.glob("report_*.html"), reverse=True):
        parts = f.stem.split("_")
        # Filename format: report_{invoice_no}_{timestamp}.html
        # invoice_no can itself contain hyphens, so we take everything except last segment
        if len(parts) >= 3:
            invoice_no = "_".join(parts[1:-1])
            timestamp = parts[-1]
        else:
            invoice_no = f.stem
            timestamp = ""
        reports.append({"file": f, "invoice_no": invoice_no, "timestamp": timestamp})
    return reports


def _infer_status(html: str) -> str:
    """Extract recommendation from report HTML content."""
    import re

    match = re.search(r'class="badge\s+(AUTO_APPROVED|MANUAL_REVIEW|REJECTED)"', html)
    if match:
        return match.group(1)
    return "UNKNOWN"


def _load_decisions() -> dict:
    if DECISIONS_LOG.exists():
        try:
            return json.loads(DECISIONS_LOG.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_decision(invoice_no: str, decision: str, reviewer: str = "operator"):
    decisions = _load_decisions()
    decisions[invoice_no] = {
        "decision": decision,
        "reviewer": reviewer,
        "timestamp": datetime.now().isoformat(),
    }
    DECISIONS_LOG.write_text(json.dumps(decisions, indent=2), encoding="utf-8")


# ============================================================================== 
# Page: Invoice Dashboard
# ============================================================================== 

def page_dashboard():
    st.header("Invoice Dashboard")

    # -- Sidebar: Run Pipeline button -----------------------------------------
    with st.sidebar:
        st.subheader("Actions")
        if st.button("Run Pipeline", use_container_width=True, type="primary"):
            with st.spinner("Running invoice pipeline..."):
                try:
                    from workflows.invoice_pipeline import run_pipeline

                    result = run_pipeline()
                    if result.get("recommendation"):
                        st.success(f"Pipeline complete: {result.get('recommendation')}")
                    else:
                        st.info("No new invoices found.")
                except Exception as exc:
                    st.error(f"Pipeline error: {exc}")

        if st.button("Clear Reports", use_container_width=True):
            removed = 0
            for report_file in REPORTS_DIR.glob("report_*.html"):
                try:
                    report_file.unlink()
                    removed += 1
                except Exception:
                    pass
            st.session_state.pop("selected_report_idx", None)
            st.info(f"Cleared {removed} report(s).")

        if st.button("Reset Processed Registry", use_container_width=True):
            try:
                from tools.invoice_watcher_tool import reset_registry

                reset_registry()
                st.info("Processed registry cleared. Invoices can be reprocessed.")
            except Exception as exc:
                st.error(f"Failed to reset registry: {exc}")

        st.divider()
        st.caption(f"Reports dir: {REPORTS_DIR}")
        st.caption(f"Incoming dir: {INCOMING_DIR}")

    # -- Load reports ----------------------------------------------------------
    reports = _load_reports()
    decisions = _load_decisions()

    if not reports:
        st.info("No reports found. Run the pipeline to process invoices.")
        return

    # -- Split layout ----------------------------------------------------------
    col_list, col_report = st.columns([1, 2.5])

    with col_list:
        st.subheader("Invoices")
        selected_idx = st.session_state.get("selected_report_idx", 0)

        for idx, r in enumerate(reports):
            html = r["file"].read_text(encoding="utf-8", errors="replace")
            status = _infer_status(html)
            emoji = STATUS_EMOJI.get(status, "?")
            decision = decisions.get(r["invoice_no"], {}).get("decision", "")
            label = f"{emoji} {r['invoice_no']}"
            if decision:
                label += f" [{decision}]"

            if st.button(label, key=f"inv_{idx}", use_container_width=True):
                st.session_state["selected_report_idx"] = idx
                selected_idx = idx

    with col_report:
        if reports:
            r = reports[selected_idx] if selected_idx < len(reports) else reports[0]
            html = r["file"].read_text(encoding="utf-8", errors="replace")
            status = _infer_status(html)

            # Status badge
            color = STATUS_COLOR.get(status, "gray")
            st.markdown(
                f"<span style='background:{color};color:white;padding:4px 12px;"
                f"border-radius:4px;font-weight:bold'>{status}</span>",
                unsafe_allow_html=True,
            )
            st.caption(f"Report: {r['file'].name}")

            # HITL buttons - only for MANUAL_REVIEW and not already decided
            existing_decision = decisions.get(r["invoice_no"], {}).get("decision", "")
            if status == "MANUAL_REVIEW" and not existing_decision:
                st.divider()
                st.markdown("**Human Review Required - make a decision:**")
                c1, c2, _ = st.columns([1, 1, 2])
                with c1:
                    if st.button("APPROVE", type="primary", key=f"approve_{r['invoice_no']}"):
                        _save_decision(r["invoice_no"], "APPROVED")
                        st.success("Decision saved: APPROVED")
                        st.rerun()
                with c2:
                    if st.button("REJECT", key=f"reject_{r['invoice_no']}"):
                        _save_decision(r["invoice_no"], "REJECTED")
                        st.warning("Decision saved: REJECTED")
                        st.rerun()
            elif existing_decision:
                st.info(f"Human decision recorded: {existing_decision}")

            st.divider()
            # Embed the HTML report
            st.components.v1.html(html, height=700, scrolling=True)


# ============================================================================== 
# Page: Invoice Q&A (RAG Chat)
# ============================================================================== 

def page_rag_chat():
    st.header("Invoice Q&A Assistant")
    st.caption("Ask questions about any processed invoice. Answers are grounded in the indexed invoice text.")

    # -- Invoice filter --------------------------------------------------------
    reports = _load_reports()
    invoice_options = ["All Invoices"] + [r["invoice_no"] for r in reports]
    selected_filter = st.selectbox("Filter by Invoice:", invoice_options)
    invoice_filter = None if selected_filter == "All Invoices" else selected_filter

    st.divider()

    # -- Chat history ----------------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "content": "Hello! Ask me anything about the processed invoices.",
                "sources": [],
                "scores": {},
            }
        ]

    # Display history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                st.caption(f"Sources: {', '.join(msg['sources'])}")
            if msg.get("scores"):
                s = msg["scores"]
                cols = st.columns(3)
                cols[0].metric("Context", f"{s.get('context_relevance', 0):.2f}")
                cols[1].metric("Grounded", f"{s.get('groundedness', 0):.2f}")
                cols[2].metric("Relevance", f"{s.get('answer_relevance', 0):.2f}")
                if s.get("low_quality"):
                    st.warning("Low quality response - consider rephrasing or indexing more invoices.")

    # -- Input -----------------------------------------------------------------
    if prompt := st.chat_input("Type your question..."):
        st.session_state["chat_history"].append(
            {"role": "user", "content": prompt, "sources": [], "scores": {}}
        )

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching invoices..."):
                try:
                    from workflows.rag_pipeline import run_rag_query

                    result = run_rag_query(prompt, invoice_no_filter=invoice_filter)
                    answer = result.get("rag_answer", "No answer generated.")
                    sources = result.get("rag_sources", [])
                    scores = result.get("rag_scores", {})
                except Exception as exc:
                    answer = f"Error running RAG query: {exc}"
                    sources = []
                    scores = {}

            st.write(answer)
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")
            if scores:
                cols = st.columns(3)
                cols[0].metric("Context", f"{scores.get('context_relevance', 0):.2f}")
                cols[1].metric("Grounded", f"{scores.get('groundedness', 0):.2f}")
                cols[2].metric("Relevance", f"{scores.get('answer_relevance', 0):.2f}")
                if scores.get("low_quality"):
                    st.warning("Low quality response - consider rephrasing or indexing more invoices.")

        st.session_state["chat_history"].append(
            {"role": "assistant", "content": answer, "sources": sources, "scores": scores}
        )


# ============================================================================== 
# App layout
# ============================================================================== 

def main():
    st.set_page_config(
        page_title="AI Invoice Auditor",
        page_icon="Invoice",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("AI Invoice Auditor")
        st.divider()
        page = st.radio(
            "Navigate",
            ["Invoice Dashboard", "Invoice Q&A"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Powered by LangGraph + ChromaDB")

    if page == "Invoice Dashboard":
        page_dashboard()
    else:
        page_rag_chat()


if __name__ == "__main__":
    main()
