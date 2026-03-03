"""
AI Invoice Auditor — Demo Runner
One-command launcher that starts all services and processes sample invoices.

Usage:
    python main.py              # Full demo (ERP + pipeline + Streamlit UI)
    python main.py --pipeline   # Run pipeline only (no UI)
    python main.py --ui         # Launch UI only (skip pipeline)
    python main.py --mcp        # Run MCP server (stdio)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_ROOT = Path(__file__).parent


def _banner():
    print("""
+----------------------------------------------------------+
|           AI Invoice Auditor  --  Demo Runner            |
|  LangGraph  ChromaDB  FastMCP  LangFuse  Streamlit       |
+----------------------------------------------------------+
""")


# ── ERP Mock Server ────────────────────────────────────────────────────────────

def start_erp_server() -> subprocess.Popen | None:
    """Start the FastAPI mock ERP server in the background."""
    erp_port = int(os.getenv("ERP_PORT", "8000"))
    print(f"[1/3] Starting Mock ERP server on port {erp_port}...")
    try:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "erp_mock.main:app",
             "--port", str(erp_port), "--log-level", "warning"],
            cwd=str(_ROOT),
        )
        time.sleep(2)  # give uvicorn a moment to boot
        print(f"      OK  ERP server running at http://localhost:{erp_port}")
        return proc
    except Exception as exc:
        print(f"      WARNING  Could not start ERP server: {exc}")
        return None


# ── Invoice Pipeline ───────────────────────────────────────────────────────────

def run_pipeline_on_all():
    """Process every unprocessed invoice in data/incoming/."""
    incoming = Path(os.getenv("INCOMING_DIR", "./data/incoming"))
    print(f"\n[2/3] Running invoice pipeline on {incoming}...")

    # Reset processed registry so all samples are reprocessed in demo mode
    registry = Path(os.getenv("PROCESSED_REGISTRY", "./data/processed_registry.json"))
    if registry.exists():
        registry.unlink()
        print("      Registry cleared -- processing all sample invoices")

    try:
        from workflows.invoice_pipeline import run_pipeline
        from core.observability import flush

        invoice_files = []
        for pattern in ("*.pdf", "*.docx", "*.png"):
            invoice_files += [
                f for f in sorted(incoming.glob(pattern))
                if not f.name.endswith(".meta.json")
            ]

        if not invoice_files:
            print("      No invoice files found in", incoming)
            return

        results = []
        for fp in invoice_files:
            print(f"      Processing {fp.name}...", end=" ", flush=True)
            try:
                state = run_pipeline(str(fp))
                rec = state.get("recommendation", "N/A")
                results.append((fp.name, rec))
                symbol = {"AUTO_APPROVED": "[OK]", "MANUAL_REVIEW": "[REVIEW]",
                          "REJECTED": "[REJECTED]"}.get(rec, "[?]")
                print(f"{symbol} {rec}")
            except Exception as exc:
                print(f"ERROR: {exc}")

        flush()

        print("\n      === Pipeline Summary ===")
        for name, rec in results:
            print(f"      {name:<35} -> {rec}")

    except Exception as exc:
        print(f"      Pipeline error: {exc}")
        raise


# ── Streamlit UI ───────────────────────────────────────────────────────────────

def launch_ui():
    """Launch the Streamlit dashboard."""
    print("\n[3/3] Launching Streamlit UI...")
    print("      Open your browser at http://localhost:8501")
    ui_path = _ROOT / "ui" / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(ui_path),
         "--server.headless", "true"],
        cwd=str(_ROOT),
    )


# ── MCP Server ─────────────────────────────────────────────────────────────────

def run_mcp_server():
    """Start the FastMCP tool registry server (stdio transport)."""
    print("Starting FastMCP server (stdio)...")
    server_path = _ROOT / "mcp_tools" / "server.py"
    subprocess.run([sys.executable, str(server_path)], cwd=str(_ROOT))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    _banner()

    args = set(sys.argv[1:])

    if "--mcp" in args:
        run_mcp_server()
        return

    erp_proc = None

    if "--ui" not in args:
        erp_proc = start_erp_server()
        try:
            run_pipeline_on_all()
        except Exception:
            pass

    if "--pipeline" not in args:
        try:
            launch_ui()
        finally:
            if erp_proc:
                erp_proc.terminate()
    else:
        if erp_proc:
            erp_proc.terminate()


if __name__ == "__main__":
    main()
