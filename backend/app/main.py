"""FastAPI application factory."""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import router
from .config import get_settings
from .db import init_db
from .logging_config import configure_logging, get_logger
from .pipeline import load_demo, run_pipeline
from .privacy import LlmPolicy

logger = get_logger(__name__)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_dirs()
    init_db()
    logger.info("digitalmind %s starting (llm_enabled=%s)", __version__,
                LlmPolicy.from_settings().enabled)
    yield
    logger.info("digitalmind shutting down")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="DigitalMind",
        version=__version__,
        description=(
            "A private, explainable AI workspace that connects your digital records. "
            "All analysis runs locally; no record content leaves this machine unless you "
            "explicitly configure an external model endpoint in Settings."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api")

    @app.get("/api/privacy-notice", tags=["meta"])
    def privacy_notice() -> JSONResponse:
        policy = LlmPolicy.from_settings()
        return JSONResponse(
            {
                "local_first": True,
                "external_transmission": policy.enabled,
                "notice": policy.status()["disclosure"],
                "integrations": "No third-party integrations are configured.",
            }
        )

    # Serve the built frontend when it exists, so `uvicorn` alone gives a full app.
    dist = settings.data_dir.parent / "frontend" / "dist"
    if dist.exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            if full_path.startswith("api/"):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            candidate = dist / full_path
            if candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app


app = create_app()
