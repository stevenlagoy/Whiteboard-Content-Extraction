"""Helpers shared by the route modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any
from urllib.parse import urlsplit

from flask import current_app, flash, redirect, request, url_for
from werkzeug.wrappers import Response

from ..container import Services
from ..domain.errors import DomainError


def services() -> Services:
    return current_app.extensions["wce"]


def safe_next(default: str) -> str:
    """The form's `next` target when it is a local path; otherwise default."""
    target = request.form.get("next", "")
    parts = urlsplit(target)
    if target.startswith("/") and not target.startswith("//") and "\\" not in target and not parts.netloc:
        return target
    return default


@dataclass(frozen=True)
class ActionResult:
    message: str | None = None
    redirect_to: str | None = None


def action(view: Callable[..., ActionResult | None]) -> Callable[..., Response]:
    """Run a POST action, flash its outcome, and redirect (Post/Redirect/Get)."""

    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any) -> Response:
        fallback = safe_next(url_for("pages.dashboard"))
        try:
            result = view(*args, **kwargs)
        except DomainError as error:
            flash(error.message, "error")
            return redirect(fallback, code=303)
        if result is not None and result.message:
            flash(result.message, "success")
        target = result.redirect_to if result is not None and result.redirect_to else fallback
        return redirect(target, code=303)

    return wrapper
