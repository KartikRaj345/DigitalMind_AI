"""Scoped logging.

Rule: record *content* is never logged. We log record ids, counts, model names and
durations. A redaction filter is installed as a belt-and-braces guard in case a
third-party library logs a payload.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

_CONFIGURED = False

# Substrings that indicate a log line may contain user content.
_SENSITIVE_HINTS = (
    "body",
    "text",
    "content",
    "prompt",
    "message_body",
    "chunk_text",
    "email",
    "subject",
)


class RedactingFilter(logging.Filter):
    """Drop user content from log records.

    We keep the message but only when it looks structural. Anything else is replaced
    with a placeholder so debugging stays possible without writing records to disk.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if not _is_redaction_enabled():
            return True
        try:
            msg = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            return True
        lowered = msg.lower()
        if any(hint in lowered for hint in _SENSITIVE_HINTS) and len(msg) > 80:
            record.msg = "[redacted: record content omitted from logs]"
            record.args = ()
        return True


def _is_redaction_enabled() -> bool:
    from .config import get_settings

    try:
        return bool(get_settings().redact_logs)
    except Exception:  # pragma: no cover - settings unavailable during teardown
        return True


def configure_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    handler.addFilter(RedactingFilter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # spaCy/thinc are extremely chatty at INFO.
    for noisy in ("thinc", "spacy", "urllib3", "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def log_counts(logger: logging.Logger, message: str, **counts: Any) -> None:
    """Structured helper: log only numeric/identifier facts, never text."""
    parts = " ".join(f"{k}={v}" for k, v in counts.items())
    logger.info("%s %s", message, parts)
