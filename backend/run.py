"""One command to run DigitalMind.

    python -m backend.run

Prepares everything that can be prepared automatically, loads the demo corpus, and starts
the API server. The built frontend in ``frontend/dist`` is served at ``/`` if it exists;
otherwise the API is on ``/api`` and the Vite dev server is the way to see the UI.

What it does, in order:
  1. checks Python 3.11 and the required packages, listing anything missing
  2. creates the data directory and database
  3. downloads the spaCy model if it is missing (skipped with ``--no-spacy-download``)
  4. loads the demo corpus and runs the full analysis pipeline, unless ``--no-demo``
  5. serves the app

Flags:
  --reset          delete the local database first
  --no-demo        start with an empty workspace
  --no-reindex     skip the embedding index (much faster startup)
  --host/--port    override the bind address
  --skip-download  never touch the network
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fail(message: str) -> None:
    print(f"\n  {message}\n", file=sys.stderr)
    raise SystemExit(1)


def check_python() -> None:
    if sys.version_info < (3, 11):
        _fail(
            f"DigitalMind needs Python 3.11 (found {sys.version.split()[0]}). "
            "The pinned dependency set is built for 3.11."
        )


def check_dependencies() -> list[str]:
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn[standard]",
        "sqlalchemy": "SQLAlchemy",
        "spacy": "spacy",
        "sklearn": "scikit-learn",
        "numpy": "numpy",
        "rank_bm25": "rank_bm25",
        "rapidfuzz": "rapidfuzz",
        "networkx": "networkx",
        "pydantic_settings": "pydantic-settings",
        "dateutil": "python-dateutil",
    }
    missing: list[str] = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    return missing


def ensure_spacy_model(model_name: str, allow_download: bool) -> str:
    try:
        import spacy

        spacy.load(model_name)
        return f"spaCy model {model_name} is installed"
    except Exception:
        pass
    if not allow_download:
        return (
            f"spaCy model {model_name} is missing. Entity extraction will use the rules "
            f"baseline only. Install it with:  python -m spacy download {model_name}"
        )
    print(f"  downloading spaCy model {model_name} (one time, ~13 MB) ...")
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", model_name], cwd=REPO_ROOT
    )
    if result.returncode != 0:
        return (
            f"could not download {model_name}; falling back to the rules baseline. "
            "Run: python -m spacy download " + model_name
        )
    return f"spaCy model {model_name} installed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run DigitalMind.")
    parser.add_argument("--reset", action="store_true", help="delete the local database first")
    parser.add_argument("--no-demo", action="store_true", help="start empty")
    parser.add_argument("--no-reindex", action="store_true", help="skip the embedding index")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true", help="never touch the network")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)

    print("\nDigitalMind")
    print("=" * 60)

    check_python()
    print(f"  python        {sys.version.split()[0]}")

    missing = check_dependencies()
    if missing:
        _fail(
            "Missing packages: "
            + ", ".join(missing)
            + "\n  Install them with:  pip install -r requirements.txt"
        )
    print("  dependencies  all present")

    from backend.app.config import get_settings

    settings = get_settings()
    settings.ensure_dirs()

    if args.reset:
        for suffix in ("", "-wal", "-shm"):
            path = settings.db_path.with_name(settings.db_path.name + suffix)
            if path.exists():
                path.unlink()
                print(f"  reset         removed {path.name}")

    from backend.app import db as db_mod

    db_mod.init_db()
    print(f"  database      {settings.db_path}")

    from backend.app.models import Record
    from sqlalchemy import func, select

    with db_mod.session_scope() as session:
        existing = int(session.execute(select(func.count()).select_from(Record)).scalar_one())

    print("  " + ensure_spacy_model(settings.spacy_model, not args.skip_download))

    if not args.no_demo and existing == 0:
        print("  demo data     loading and analysing (this takes ~10-30 seconds) ...")
        from backend.app.pipeline import load_demo, run_pipeline

        with db_mod.session_scope() as session:
            stats = load_demo(session)
            summary = run_pipeline(session, with_embeddings=not args.no_reindex)
        print(
            f"  demo data     {stats.inserted} records, "
            f"{summary['resolution']['entities']} entities, "
            f"{summary['insights']['created']} insights"
        )
    elif existing:
        print(f"  records       {existing} already imported (use --reset for a clean start)")
    else:
        print("  records       empty workspace; use the Import screen to add your own")

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        heavy = "available (torch + transformers)"
    except Exception:
        heavy = "not installed - rules + TF-IDF detection only (pip install -r requirements-ml.txt)"
    print(f"  ml extras     {heavy}")

    dist = REPO_ROOT / "frontend" / "dist" / "index.html"
    if dist.exists():
        print(f"  frontend      serving {dist.parent.relative_to(REPO_ROOT)} at /")
    else:
        print(
            "  frontend      not built. Run 'npm install && npm run build' in frontend/, "
            "or use 'npm run dev' for the Vite dev server on :5173"
        )

    host = args.host or settings.host
    port = args.port or settings.port

    from backend.app.privacy import LlmPolicy

    policy = LlmPolicy.from_settings()
    print("\n  privacy       all analysis runs locally; no record content leaves this machine.")
    if not policy.enabled:
        print("                the optional external LLM layer is off (the default).")
    print("\n  open          http://%s:%d\n" % (host, port))

    import uvicorn

    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
