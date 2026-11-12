"""Flask application factory: wires configuration, services, routes, and assets together."""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import Executor
from datetime import datetime

from flask import Flask, jsonify, render_template, request
from flask_assets import Bundle, Environment  # type: ignore
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .config import AppConfig
from .container import Services
from .demo_data import seed_demo_library
from .domain.errors import ConflictError, DomainError, NotFoundError
from .presentation import format_bytes, register_presentation
from .processing.base import ProcessingEngine
from .processing.registry import build_engine
from .processing.runner import JobRunner
from .repositories.base import Repository
from .repositories.memory import InMemoryRepository
from .routes.actions import actions
from .routes.files import files
from .routes.pages import pages
from .services.exports import ExportService
from .services.ingest import IngestService
from .services.jobs import JobService
from .services.library import LibraryService
from .services.media import MediaProbe, OpenCVMediaProbe
from .services.status import StatusService
from .storage import MediaStorage


def create_app(
    config: AppConfig | None = None,
    *,
    engine: ProcessingEngine | None = None,
    media_probe: MediaProbe | None = None,
    repository: Repository | None = None,
    clock: Callable[[], datetime] | None = None,
    executor: Executor | None = None,
) -> Flask:
    """Build the app. Every keyword argument replaces one part of the system."""
    config = config or AppConfig.from_env()
    clock = clock or datetime.now
    repo = repository or InMemoryRepository()
    engine = engine or build_engine(config)

    storage = MediaStorage(config.data_dir)
    library = LibraryService(repo, clock)
    jobs = JobService(library, repo, engine.estimate_seconds, clock, config.max_concurrent_jobs)
    services = Services(
        config=config,
        clock=clock,
        storage=storage,
        library=library,
        jobs=jobs,
        status=StatusService(library, repo, engine.name, clock),
        ingest=IngestService(library, jobs, media_probe or OpenCVMediaProbe(), storage, clock),
        exports=ExportService(library),
        runner=JobRunner(jobs, library, engine, storage, executor, config.max_concurrent_jobs),
    )

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=config.secret_key,
        MAX_CONTENT_LENGTH=config.max_upload_bytes,
        LIBSASS_INCLUDE_PATHS=[os.path.join(app.root_path, "static", "scss")],
    )
    app.extensions["wce"] = services

    _register_assets(app)
    register_presentation(app, services)
    for blueprint in (pages, actions, files):
        app.register_blueprint(blueprint)
    _register_error_handlers(app, config)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}, 200

    if config.seed_demo and not repo.folders():
        seed_demo_library(library, repo, clock())
    if config.start_runner:
        app.before_request(services.runner.ensure_started)
    return app


def _register_assets(app: Flask) -> None:
    assets = Environment(app)
    assets.register(
        "scss_all",
        Bundle(
            "scss/main.scss",
            filters="libsass",
            output="css/packed.css",
            depends=("scss/*.scss",),  # rebuild when any partial changes
        ),
    )


def _register_error_handlers(app: Flask, config: AppConfig) -> None:
    def respond(status: int, title: str, message: str):
        if request.accept_mimetypes.best == "application/json":
            return jsonify(error=message), status
        return render_template("error.html", status=status, title=title, message=message), status

    @app.errorhandler(DomainError)
    def domain_error(error: DomainError):
        status = 404 if isinstance(error, NotFoundError) else 409 if isinstance(error, ConflictError) else 400
        return respond(status, "That did not work", error.message)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error: RequestEntityTooLarge):
        return respond(413, "File too large", f"Uploads can be up to {format_bytes(config.max_upload_bytes)}.")

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        title = "Page not found" if error.code == 404 else error.name
        message = "That page does not exist." if error.code == 404 else (error.description or error.name)
        return respond(error.code or 500, title, message)


if __name__ == "__main__":
    create_app().run(debug=True)
