"""
LangFuse Observability — AI Invoice Auditor
Provides a @trace_agent decorator that wraps LangGraph node functions with
LangFuse traces and spans for full pipeline observability.

If LANGFUSE_SECRET_KEY / LANGFUSE_PUBLIC_KEY are not set (or are placeholder
values starting with "your_"), tracing is silently disabled — the pipeline
runs normally without any observability overhead.

Usage:
    from core.observability import trace_agent

    @trace_agent("extractor")
    def extractor_agent(state: InvoiceState) -> dict:
        ...
"""

from __future__ import annotations

import os
from functools import wraps
from typing import Callable

from core.logger import get_logger

logger = get_logger(__name__)

# ── LangFuse client (lazy singleton) ──────────────────────────────────────────

_langfuse = None
_tracing_enabled: bool | None = None  # None = not yet checked


def _get_langfuse():
    """Return a live Langfuse client or None if credentials are missing."""
    global _langfuse, _tracing_enabled

    if _tracing_enabled is False:
        return None
    if _langfuse is not None:
        return _langfuse

    secret_key = os.getenv("LANGFUSE_SECRET_KEY", "")
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "")

    if not secret_key or secret_key.startswith("your_") or not public_key or public_key.startswith("your_"):
        logger.info("LangFuse credentials not configured — tracing disabled")
        _tracing_enabled = False
        return None

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        _tracing_enabled = True
        logger.info("LangFuse tracing enabled (host=%s)", os.getenv("LANGFUSE_HOST"))
        return _langfuse
    except ImportError:
        logger.warning("langfuse package not installed — tracing disabled")
        _tracing_enabled = False
        return None
    except Exception as exc:
        logger.warning("LangFuse initialisation failed: %s — tracing disabled", exc)
        _tracing_enabled = False
        return None


# ── Decorator ─────────────────────────────────────────────────────────────────

def trace_agent(agent_name: str) -> Callable:
    """
    Decorator that wraps a LangGraph node function with a LangFuse trace + span.

    Args:
        agent_name: Human-readable name for the agent (used as span name).

    Example:
        @trace_agent("extractor")
        def extractor_agent(state: InvoiceState) -> dict:
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(state: dict) -> dict:
            lf = _get_langfuse()
            if lf is None:
                return func(state)

            invoice_path = state.get("file_path", "unknown")
            invoice_no = (state.get("extracted_fields") or {}).get("invoice_no", "")

            try:
                trace = lf.trace(
                    name=f"invoice_pipeline.{agent_name}",
                    input={"file_path": invoice_path, "invoice_no": invoice_no},
                    tags=["invoice_auditor", agent_name],
                )
                span = trace.span(
                    name=agent_name,
                    input={
                        "file_path": invoice_path,
                        "detected_language": state.get("detected_language", ""),
                        "recommendation": state.get("recommendation", ""),
                    },
                )
            except Exception as exc:
                logger.debug("LangFuse span creation failed: %s", exc)
                return func(state)

            try:
                result = func(state)
                try:
                    span.end(
                        output={
                            "keys_updated": list(result.keys()),
                            "errors": result.get("errors", []),
                        }
                    )
                    lf.flush()
                except Exception:
                    pass
                return result

            except Exception as exc:
                try:
                    span.end(
                        output={"error": str(exc)},
                        level="ERROR",
                    )
                    lf.flush()
                except Exception:
                    pass
                raise

        return wrapper
    return decorator


# ── Convenience flush ─────────────────────────────────────────────────────────

def flush():
    """Flush any pending LangFuse events. Call at pipeline end."""
    lf = _get_langfuse()
    if lf is not None:
        try:
            lf.flush()
        except Exception:
            pass
