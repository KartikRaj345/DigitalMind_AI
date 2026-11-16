"""Test fixtures.

Every test gets an isolated on-disk SQLite database so nothing leaks between tests and
the cascade-delete test can inspect the real schema.
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from backend.app import db as db_mod
from backend.app.config import reset_settings_cache


@pytest.fixture()
def temp_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    tmp = Path(tempfile.mkdtemp(prefix="digitalmind-test-"))
    monkeypatch.setenv("DIGITALMIND_DATA_DIR", str(tmp))
    monkeypatch.setenv("DIGITALMIND_DATABASE_URL", f"sqlite:///{tmp / 'test.db'}")
    monkeypatch.setenv("DIGITALMIND_ARTIFACTS_DIR", str(tmp / "artifacts"))
    monkeypatch.setenv("DIGITALMIND_REDACT_LOGS", "true")
    reset_settings_cache()
    db_mod.dispose_engine()
    yield tmp
    db_mod.dispose_engine()
    reset_settings_cache()


@pytest.fixture()
def session(temp_env: Path):
    db_mod.init_db()
    with db_mod.session_scope() as s:
        yield s


@pytest.fixture()
def client(temp_env: Path):
    from fastapi.testclient import TestClient

    from backend.app.main import create_app

    db_mod.init_db()
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def demo_session(session):
    """A session with the demo corpus ingested and fully analysed."""
    from backend.app.pipeline import load_demo, run_pipeline

    load_demo(session)
    run_pipeline(session, with_embeddings=False)
    return session


def utc(*args) -> dt.datetime:
    return dt.datetime(*args, tzinfo=dt.timezone.utc)
