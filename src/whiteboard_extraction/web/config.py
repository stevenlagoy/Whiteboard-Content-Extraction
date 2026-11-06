"""Settings for the web app, read from environment variables."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

_FALSE = {"0", "false", "no", "off", ""}


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in _FALSE


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path = Path("instance")
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))
    engine: str = "simulated"
    recognizer: str = "ocr"
    llm_model: str = ""
    cnn_checkpoint: str = ""
    seed_demo: bool = True
    start_runner: bool = True
    max_upload_bytes: int = 4 * 1024**3
    max_concurrent_jobs: int = 1
    recent_limit: int = 8

    @classmethod
    def from_env(cls) -> AppConfig:
        env = os.environ
        return cls(
            data_dir=Path(env.get("UPLOAD_ROOT", "instance")),
            secret_key=env.get("SECRET_KEY") or secrets.token_hex(32),
            engine=env.get("WCE_ENGINE", "simulated"),
            recognizer=env.get("WCE_RECOGNIZER", "ocr"),
            llm_model=env.get("WCE_LLM_MODEL", ""),
            cnn_checkpoint=env.get("WCE_CNN_CHECKPOINT", ""),
            seed_demo=_flag("WCE_SEED_DEMO", True),
            max_upload_bytes=int(float(env.get("WCE_MAX_UPLOAD_GB", "4")) * 1024**3),
            max_concurrent_jobs=int(env.get("WCE_MAX_CONCURRENT_JOBS", "1")),
        )
