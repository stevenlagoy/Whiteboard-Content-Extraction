"""The object graph shared by every request. Built once in create_app."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from .config import AppConfig
from .processing.runner import JobRunner
from .services.exports import ExportService
from .services.ingest import IngestService
from .services.jobs import JobService
from .services.library import LibraryService
from .services.status import StatusService
from .storage import MediaStorage


@dataclass(frozen=True)
class Services:
    config: AppConfig
    clock: Callable[[], datetime]
    storage: MediaStorage
    library: LibraryService
    jobs: JobService
    status: StatusService
    ingest: IngestService
    exports: ExportService
    runner: JobRunner
