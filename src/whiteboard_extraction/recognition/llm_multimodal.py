"""Recognition backend using a multimodal LLM.

Two transports are built in, selected by ``provider`` (or the ``WCE_LLM_PROVIDER`` variable):

* ``"anthropic"`` (default): the Anthropic Claude API through the ``anthropic`` SDK.
* ``"openai"``: any server that speaks the OpenAI chat-completions format. That covers a local
  Ollama (``http://localhost:11434/v1``, no key) and hosted endpoints such as GitHub Models,
  OpenRouter, Groq and Google AI Studio (a base URL plus a key).

Both are injectable for tests: ``client=`` is an SDK-like object with ``messages.create(...)``,
``http_client=`` is an object with ``post(url, headers, payload, timeout) -> dict``. The
``anthropic`` SDK is imported lazily so the package imports without it; the OpenAI-style
transport uses only the standard library.

Configuration (constructor arguments win over environment variables)::

    WCE_LLM_PROVIDER     anthropic | openai                 default: anthropic
    WCE_LLM_MODEL        model id used when none is passed  default: per provider, see DEFAULT_MODELS
    WCE_LLM_BASE_URL     endpoint base URL                  default: http://localhost:11434/v1 for openai
    WCE_LLM_API_KEY_ENV  NAME of the variable holding the key   default: ANTHROPIC_API_KEY for anthropic,
                                                            none for openai (no Authorization header)
    WCE_LLM_TIMEOUT      request timeout in seconds         default: 600
    WCE_LLM_REASONING_EFFORT  (openai) reasoning_effort to send; default "auto": "low" for local servers
                         such as Ollama, nothing for hosted ones; "omit" never sends it
    WCE_LLM_JSON_SCHEMA  (openai) send a JSON-schema response_format: "auto" (default: local servers only),
                         "on" or "off"
    WCE_LLM_EXTRA_BODY   (openai) JSON object merged into the request body, for provider-specific fields

Usage::

    recognizer = MultimodalLLMRecognizer("")          # "" -> the provider's default model
    blocks = recognizer.recognize(12.5, bgr_keyframe)  # list[RecognizedBlock]

Try it on one image::

    python -m whiteboard_extraction.recognition.llm_multimodal path/to/board.png
    python -m whiteboard_extraction.recognition.llm_multimodal --provider openai path/to/board.png
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import cv2
import numpy as np
from PIL import Image

from whiteboard_extraction.recognition.base import RecognizedBlock

PROVIDERS = ("anthropic", "openai")
DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-opus-5-5"  # default for the anthropic provider
# Ollama tag of a small vision model that fits an 8 GB laptop. The "-instruct" variant does not think;
# the bare "qwen3-vl:2b" / "qwen3-vl:4b" tags are the thinking variants, which fill Ollama's default
# 4k context window with reasoning before answering.
DEFAULT_OPENAI_MODEL = "qwen3-vl:2b-instruct"
DEFAULT_MODELS = {"anthropic": DEFAULT_MODEL, "openai": DEFAULT_OPENAI_MODEL}
DEFAULT_OPENAI_BASE_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT = 600.0

API_KEY_ENV = "ANTHROPIC_API_KEY"
PROVIDER_ENV = "WCE_LLM_PROVIDER"
MODEL_ENV = "WCE_LLM_MODEL"
BASE_URL_ENV = "WCE_LLM_BASE_URL"
KEY_NAME_ENV = "WCE_LLM_API_KEY_ENV"
TIMEOUT_ENV = "WCE_LLM_TIMEOUT"
REASONING_EFFORT_ENV = "WCE_LLM_REASONING_EFFORT"
JSON_SCHEMA_ENV = "WCE_LLM_JSON_SCHEMA"
EXTRA_BODY_ENV = "WCE_LLM_EXTRA_BODY"
# Thinking models (Qwen3-VL on Ollama, for example) can spend the whole context window on reasoning
# and return no answer. Local servers get this default; "omit" sends nothing.
DEFAULT_LOCAL_REASONING_EFFORT = "low"

# The reply shape, as a JSON schema for servers that support ``response_format`` (Ollama does).
# Small local models otherwise drift to a bare list of strings now and then.
BLOCKS_JSON_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {"text": {"type": "string"}, "kind": {"type": "string", "enum": ["prose", "equation"]}},
        "required": ["text", "kind"],
    },
}
RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {"name": "board_blocks", "schema": BLOCKS_JSON_SCHEMA},
}

# Long-edge limit of the high-resolution vision tier (Claude 4.7 and later). The API
# downscales anything larger itself, so sending more pixels only costs upload time.
MAX_IMAGE_SIDE = 2576
VALID_KINDS = ("prose", "equation")

SYSTEM_PROMPT = """\
You transcribe photographs of lecture whiteboards and blackboards into structured notes for \
students who cannot see the board.

Return ONLY a JSON array. No prose before or after it, no Markdown code fences. Each element is an \
object with exactly two keys:
  "text": the content of one block, as a string
  "kind": "prose" or "equation"

Rules:
- List blocks in the order a reader would read the board: top to bottom, and left to right when the \
board is divided into columns.
- Use "equation" for any line that contains mathematics: an equals sign, an inequality, a fraction, an \
exponent or subscript, a function value such as S(3) or f(x), a sum or a product, even when a few words \
accompany it (for example "Assume true for k: S(k) = k(k+1)/2" is an equation). Write the "text" of an \
equation as LaTeX without surrounding $ delimiters, for example "\\\\frac{dy}{dx} = 2x" (backslashes \
doubled, as JSON strings require).
- Use "prose" only for lines made of words with no mathematics: sentences, headings, bullet points, labels \
and code. Write it as plain text.
- One block per written line or expression, and each written line exactly once: never repeat a block. A \
sentence that wraps across lines is one prose block; a multi-line derivation is one equation block per line.
- Transcribe what is written. Do not correct, complete, or comment on the content, and do not describe \
drawings, diagrams, arrows, or the room. Faint erased marks are not content.
- If the board is blank or contains nothing legible, return [].

Example of the exact output shape: \
[{"text": "Chain rule", "kind": "prose"}, {"text": "\\\\frac{dy}{dx} = 2x", "kind": "equation"}]
"""

USER_PROMPT = "Transcribe this board. Respond with the JSON array only."


# --- image encoding -------------------------------------------------------------------------------

def encode_image_png(image: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> tuple[str, tuple[int, int]]:
    """Convert a BGR (OpenCV) image to RGB PNG, downscaled so no side exceeds ``max_side``.

    Returns the base64-encoded PNG and the ``(width, height)`` that was actually encoded.
    Grayscale (H x W) and BGRA (H x W x 4) inputs are accepted too.
    """
    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3) or image.size == 0:
        raise ValueError("image must be a non-empty numpy array of shape (H, W), (H, W, 3) or (H, W, 4)")
    if max_side < 1:
        raise ValueError("max_side must be a positive integer")

    array = np.ascontiguousarray(image)
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)

    if array.ndim == 2:
        rgb = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    elif array.shape[2] == 3:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    elif array.shape[2] == 4:
        rgb = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
    else:
        raise ValueError(f"unsupported channel count {array.shape[2]}; expected 1, 3 or 4")

    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest > max_side:
        scale = max_side / longest
        width = max(1, round(width * scale))
        height = max(1, round(height * scale))
        rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)

    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("ascii"), (width, height)


# --- reply parsing --------------------------------------------------------------------------------

def _excerpt(text: str, limit: int = 200) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[:limit] + "..."


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_-]*\s*(.*?)```", re.DOTALL)
_THINK_RE = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.DOTALL | re.IGNORECASE)


def split_think(text: str) -> tuple[str, str]:
    """Separate inline ``<think>...</think>`` reasoning (an unterminated tag runs to the end) from the answer."""
    reasoning = "\n".join(match.group(1).strip() for match in _THINK_RE.finditer(text))
    return reasoning, _THINK_RE.sub("", text)


def parse_blocks(reply: str, collapse_repeats: bool = True) -> list[dict[str, Any]]:
    """Parse the model's reply into validated ``{"text", "kind"[, "confidence"]}`` dicts.

    Tolerates Markdown code fences and stray prose around the array. Raises ``RuntimeError``
    (with an excerpt of the reply) when there is no JSON array, when an element is malformed,
    or when a ``kind`` is not one of ``VALID_KINDS``. Blocks with empty text are dropped, and
    with ``collapse_repeats`` a block that repeats the previous one is dropped too: small models
    sometimes loop on a formula and emit it five times.
    """
    if not isinstance(reply, str):
        raise RuntimeError(f"LLM reply is not text: {type(reply).__name__}")

    data = _extract_json(reply)
    if isinstance(data, dict) and isinstance(data.get("blocks"), list):
        data = data["blocks"]
    if not isinstance(data, list):
        raise RuntimeError(f"LLM reply is not a JSON array: {_excerpt(reply)!r}")

    validated = (_validate_block(index, item, reply) for index, item in enumerate(data))
    blocks = [block for block in validated if block is not None]
    return _collapse_repeats(blocks) if collapse_repeats else blocks


def _collapse_repeats(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop blocks that repeat the previous block (same kind, same text up to whitespace)."""
    kept: list[dict[str, Any]] = []
    for block in blocks:
        previous = kept[-1] if kept else None
        if previous and previous["kind"] == block["kind"] and previous["text"].split() == block["text"].split():
            continue
        kept.append(block)
    return kept


def _extract_json(reply: str) -> Any:
    """Find the JSON payload in a reply: fenced block first, then the whole text, then the outermost [...]."""
    candidate = split_think(reply)[1].strip()
    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    data = _load_json(candidate)
    if data is None:
        start, end = candidate.find("["), candidate.rfind("]")
        if start != -1 and end > start:
            data = _load_json(candidate[start:end + 1])
    if data is None:
        raise RuntimeError(f"LLM reply is not valid JSON: {_excerpt(reply)!r}")
    return data


def _validate_block(index: int, item: Any, reply: str) -> dict[str, Any] | None:
    """Check one array element; return the cleaned block, or None when its text is empty."""
    if not isinstance(item, dict):
        raise RuntimeError(f"LLM reply element {index} is not an object: {_excerpt(reply)!r}")
    text = item.get("text")
    kind = item.get("kind")
    if not isinstance(text, str):
        raise RuntimeError(f"LLM reply element {index} has no string 'text': {_excerpt(reply)!r}")
    if not isinstance(kind, str) or kind.strip().lower() not in VALID_KINDS:
        raise RuntimeError(
            f"LLM reply element {index} has invalid kind {kind!r} (expected one of {VALID_KINDS}): "
            f"{_excerpt(reply)!r}"
        )
    text = text.strip()
    if not text:
        return None
    block: dict[str, Any] = {"text": text, "kind": kind.strip().lower()}
    confidence = item.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        block["confidence"] = float(confidence)
    return block


def _load_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def _field(block: Any, name: str) -> Any:
    """Read ``name`` from an SDK content block (attribute) or a plain dict."""
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def _env(name: str) -> str | None:
    """A non-empty environment variable, stripped, or None."""
    value = os.environ.get(name, "").strip()
    return value or None


def _read_api_key(api_key_env: str | None, hint: str = "") -> str | None:
    """Return the key named by ``api_key_env``; None when no variable is configured; error when it is empty."""
    if not api_key_env:
        return None
    key = os.environ.get(api_key_env, "").strip()
    if not key:
        raise RuntimeError(
            f"No API key for the multimodal LLM backend: set the {api_key_env} environment variable{hint}."
        )
    return key


# --- transports -----------------------------------------------------------------------------------

class AnthropicTransport:
    """Calls the Anthropic Messages API through the ``anthropic`` SDK (imported on first use)."""

    def __init__(
        self,
        model: str,
        *,
        client: Any = None,
        api_key_env: str = API_KEY_ENV,
        base_url: str | None = None,
        max_tokens: int = 8192,
        effort: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.model = model
        self.api_key_env = api_key_env or API_KEY_ENV
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.effort = effort
        self.timeout = timeout
        self._client = client

    def build_request(self, png_b64: str) -> dict[str, Any]:
        """Keyword arguments for ``messages.create``: image first, then the instruction."""
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            }],
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}
        return request

    def complete(self, png_b64: str) -> str:
        """Send one image and return the reply text. Every failure is a ``RuntimeError``."""
        request = self.build_request(png_b64)
        client = self._get_client()
        try:
            response = client.messages.create(**request)
        except Exception as exc:  # SDK errors (auth, rate limit, network, ...) and anything else
            raise RuntimeError(f"Multimodal LLM request failed ({type(exc).__name__}): {exc}") from exc
        return self._response_text(response)

    def _get_client(self) -> Any:
        if self._client is None:
            api_key = _read_api_key(self.api_key_env, ", or pass client=... to MultimodalLLMRecognizer")
            try:
                import anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "The 'anthropic' package is required for the anthropic provider: pip install anthropic"
                ) from exc
            options: dict[str, Any] = {"api_key": api_key, "timeout": self.timeout}
            if self.base_url:
                options["base_url"] = self.base_url
            self._client = anthropic.Anthropic(**options)
        return self._client

    @staticmethod
    def _response_text(response: Any) -> str:
        stop_reason = _field(response, "stop_reason")
        if stop_reason == "refusal":
            details = _field(response, "stop_details")
            explanation = _field(details, "explanation") if details is not None else None
            raise RuntimeError(f"Multimodal LLM refused the request: {explanation or 'no explanation given'}")

        content = _field(response, "content") or []
        text = "".join(_field(block, "text") or "" for block in content if _field(block, "type") == "text")
        if stop_reason == "max_tokens":
            raise RuntimeError(
                f"Multimodal LLM reply was cut off at max_tokens; raise max_tokens. Partial reply: {_excerpt(text)!r}"
            )
        if not text.strip():
            raise RuntimeError(f"Multimodal LLM returned no text (stop_reason={stop_reason!r})")
        return text


def _is_local_url(url: str) -> bool:
    return urllib.parse.urlparse(url).hostname in ("localhost", "127.0.0.1", "::1")


class UrllibHTTPClient:
    """Minimal JSON-over-HTTP POST on the standard library.

    Failures raise ``RuntimeError`` carrying the HTTP status and the server's reply excerpt, so a
    404 from Ollama ("model not found, try pulling it first") reaches the user verbatim.
    """

    def post(self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float | None) -> Any:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")
            raise RuntimeError(f"HTTP {exc.code} from {url}: {_excerpt(detail)}") from exc
        except urllib.error.URLError as exc:
            hint = " (is the server running? for Ollama: ollama serve)" if _is_local_url(url) else ""
            raise RuntimeError(f"could not reach {url}: {exc.reason}{hint}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"no reply from {url} within {timeout} s; raise the timeout for slow models") from exc
        except OSError as exc:  # connection reset, broken pipe, ...
            raise RuntimeError(f"connection to {url} failed: {exc}") from exc
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"non-JSON reply from {url}: {_excerpt(raw.decode('utf-8', 'replace'))!r}") from exc


class OpenAICompatibleTransport:
    """POSTs to ``{base_url}/chat/completions`` in the OpenAI chat-completions format.

    Works with Ollama (no key) and with hosted OpenAI-style endpoints (GitHub Models, OpenRouter,
    Groq, Google AI Studio) when ``api_key_env`` names the variable that holds the key.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key_env: str | None = None,
        max_tokens: int = 8192,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: Any = None,
        reasoning_effort: str | None = None,
        json_schema: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ):
        self.model = model
        self.base_url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self.url = self.base_url + "/chat/completions"
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.http_client = http_client if http_client is not None else UrllibHTTPClient()
        self.extra_body = dict(extra_body or {})
        # Fields chosen automatically for local servers are dropped (once, then remembered) when the
        # server rejects them; explicitly configured ones are not.
        self.auto_fields: set[str] = set()
        self.reasoning_effort = self._resolve_reasoning_effort(reasoning_effort)
        self.response_format = self._resolve_response_format(json_schema)

    def _resolve_reasoning_effort(self, value: str | None) -> str | None:
        """``None``/``"auto"`` -> the local default for local servers, nothing for hosted ones; ``"omit"`` -> nothing."""
        value = (value or "auto").strip().lower()
        if value == "auto":
            self.auto_fields.add("reasoning_effort")
            return DEFAULT_LOCAL_REASONING_EFFORT if _is_local_url(self.url) else None
        return None if value in ("omit", "") else value

    def _resolve_response_format(self, value: str | None) -> dict[str, Any] | None:
        """``"auto"`` -> the blocks schema for local servers only; ``"on"`` always; ``"off"`` never."""
        value = (value or "auto").strip().lower()
        if value == "auto":
            self.auto_fields.add("response_format")
            return RESPONSE_FORMAT if _is_local_url(self.url) else None
        if value in ("on", "true", "1", "yes"):
            return RESPONSE_FORMAT
        if value in ("off", "false", "0", "no", "omit", ""):
            return None
        raise ValueError(f"{JSON_SCHEMA_ENV} must be auto, on or off, got {value!r}")

    def build_request(self, png_b64: str) -> dict[str, Any]:
        """The JSON body: system prompt, then a user turn with the image (data URL) before the instruction.

        ``reasoning_effort`` is added when configured and ``extra_body`` is merged last, so it can
        override anything (for example ``{"temperature": 0}`` or a provider's own thinking switch).
        """
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_b64}"}},
                        {"type": "text", "text": USER_PROMPT},
                    ],
                },
            ],
        }
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort
        if self.response_format:
            request["response_format"] = self.response_format
        request.update(self.extra_body)
        return request

    def headers(self) -> dict[str, str]:
        """JSON headers plus ``Authorization: Bearer`` when a key variable is configured (read at call time)."""
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        api_key = _read_api_key(self.api_key_env, f" (named by {KEY_NAME_ENV} or api_key_env)")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def complete(self, png_b64: str) -> str:
        """Send one image and return the reply text. Every failure is a ``RuntimeError``."""
        headers = self.headers()
        payload = self.build_request(png_b64)
        try:
            data = self._post(headers, payload)
        except RuntimeError as exc:
            # A model without thinking support rejects the automatic reasoning_effort (Ollama: HTTP 400
            # '"<model>" does not support thinking'); a server without structured output rejects the
            # response_format. Drop the rejected automatic field(s) for this model and try once more.
            rejected = [field for field in self.auto_fields if field in payload and _rejects_field(exc, field)]
            if not rejected:
                raise
            for field in rejected:
                self.auto_fields.discard(field)
                setattr(self, field, None)
            data = self._post(headers, {key: value for key, value in payload.items() if key not in rejected})
        return self._response_text(data)

    def _post(self, headers: dict[str, str], payload: dict[str, Any]) -> Any:
        try:
            return self.http_client.post(self.url, headers, payload, self.timeout)
        except RuntimeError as exc:
            message = f"Multimodal LLM request failed: {exc}"
            if "not found" in str(exc) and _is_local_url(self.url):
                message += f" (pull it first: ollama pull {self.model})"
            raise RuntimeError(message) from exc
        except Exception as exc:
            raise RuntimeError(f"Multimodal LLM request failed ({type(exc).__name__}): {exc}") from exc

    @staticmethod
    def _response_text(data: Any) -> str:
        if not isinstance(data, dict):
            raise RuntimeError(f"Multimodal LLM returned an unexpected reply: {_excerpt(str(data))!r}")
        if data.get("error"):
            raise RuntimeError(f"Multimodal LLM API error: {_excerpt(json.dumps(data['error']))}")
        choices = data.get("choices") or []
        if not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"Multimodal LLM returned no choices: {_excerpt(json.dumps(data))!r}")

        choice = choices[0]
        text, reasoning = _message_text(choice.get("message"))
        finish = choice.get("finish_reason")
        if finish == "content_filter":
            raise RuntimeError("Multimodal LLM refused the request (finish_reason=content_filter)")
        if not text.strip() and reasoning:
            raise RuntimeError(
                f"Multimodal LLM spent its reply on reasoning ({len(reasoning)} characters) and returned no answer "
                f"(finish_reason={finish!r}). Lower {REASONING_EFFORT_ENV} (for example 'low' or 'none'), give the "
                f"server a larger context window, or raise max_tokens. Reasoning starts: {_excerpt(reasoning, 120)!r}"
            )
        if finish == "length":
            raise RuntimeError(
                f"Multimodal LLM reply was cut off at max_tokens; raise max_tokens. Partial reply: {_excerpt(text)!r}"
            )
        if not text.strip():
            raise RuntimeError(f"Multimodal LLM returned no text (finish_reason={finish!r})")
        return text


_FIELD_COMPLAINTS = {
    "reasoning_effort": ("reasoning", "thinking"),
    "response_format": ("response_format", "json_schema", "structured", "format"),
}


def _rejects_field(exc: Exception, field: str) -> bool:
    """True for a client error (HTTP 4xx) whose message complains about ``field``."""
    text = str(exc).lower()
    return "http 4" in text and any(word in text for word in _FIELD_COMPLAINTS[field])


def _message_text(message: Any) -> tuple[str, str]:
    """``(answer, reasoning)`` of a chat-completions message.

    ``content`` may be a string or a list of parts. Reasoning comes from a ``reasoning`` or
    ``reasoning_content`` field (Ollama, OpenRouter, vLLM) or from inline ``<think>`` tags.
    """
    if not isinstance(message, dict):
        return "", ""
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict) and part.get("type") == "text"
        )
    text = content if isinstance(content, str) else ""
    inline, text = split_think(text)
    field = message.get("reasoning") or message.get("reasoning_content") or ""
    reasoning = "\n".join(part for part in (str(field).strip(), inline.strip()) if part)
    return text, reasoning


# --- the recognizer -------------------------------------------------------------------------------

class MultimodalLLMRecognizer:
    """Sends a keyframe image to a multimodal LLM and parses its response.

    Args:
        model: Model id, stored as ``self.model``. Empty -> ``WCE_LLM_MODEL`` -> the provider's default.
        provider: ``"anthropic"`` or ``"openai"``. Empty -> ``WCE_LLM_PROVIDER`` -> ``"anthropic"``.
        client: (anthropic) a pre-built SDK client, or any object with a compatible ``messages.create``.
        http_client: (openai) an object with ``post(url, headers, payload, timeout) -> dict``.
        base_url: Endpoint base URL. Empty -> ``WCE_LLM_BASE_URL`` -> ``http://localhost:11434/v1`` for
            openai, the SDK default for anthropic.
        api_key_env: NAME of the environment variable holding the key. Empty -> ``WCE_LLM_API_KEY_ENV``
            -> ``ANTHROPIC_API_KEY`` for anthropic, none for openai (Ollama needs no key).
        max_tokens: Output token cap for one board. Raise it if ``recognize`` reports truncation.
        max_image_side: Longest image side sent, in pixels. Lower it (for example 1280) to cut image
            tokens and, on a local model, time per frame.
        effort: Optional ``output_config.effort`` level for the anthropic provider only.
        timeout: Seconds to wait for one reply. Empty -> ``WCE_LLM_TIMEOUT`` -> 600.
        reasoning_effort: (openai) ``reasoning_effort`` to send. Empty -> ``WCE_LLM_REASONING_EFFORT`` ->
            ``"auto"``: ``"low"`` for local servers such as Ollama, nothing for hosted ones. ``"omit"`` never
            sends it. Thinking models otherwise fill the context window with reasoning and return nothing.
        json_schema: (openai) send the blocks JSON schema as ``response_format``: ``"auto"`` (default: local
            servers only, where small models otherwise drift from the object shape), ``"on"`` or ``"off"``.
            Empty -> ``WCE_LLM_JSON_SCHEMA``.
        extra_body: (openai) fields merged into the request body last. Empty -> ``WCE_LLM_EXTRA_BODY`` (JSON).
        collapse_repeats: Drop a block that repeats the previous one (small models loop on a formula). Default on.
    """

    def __init__(
        self,
        model: str = "",
        *,
        provider: str | None = None,
        client: Any = None,
        http_client: Any = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        max_tokens: int = 8192,
        max_image_side: int = MAX_IMAGE_SIDE,
        effort: str | None = None,
        timeout: float | None = None,
        reasoning_effort: str | None = None,
        json_schema: str | None = None,
        extra_body: dict[str, Any] | None = None,
        collapse_repeats: bool = True,
    ):
        provider = (provider or _env(PROVIDER_ENV) or DEFAULT_PROVIDER).strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"unknown multimodal LLM provider {provider!r}; choose one of {', '.join(PROVIDERS)}")
        self.provider = provider
        self.model = (model or "").strip() or _env(MODEL_ENV) or DEFAULT_MODELS[provider]
        self.max_tokens = max_tokens
        self.max_image_side = max_image_side
        self.effort = effort
        self.collapse_repeats = collapse_repeats
        self.timeout = _resolve_timeout(timeout)

        base_url = base_url or _env(BASE_URL_ENV)
        api_key_env = api_key_env or _env(KEY_NAME_ENV)
        if provider == "anthropic":
            self.transport: Any = AnthropicTransport(
                self.model, client=client, api_key_env=api_key_env or API_KEY_ENV, base_url=base_url,
                max_tokens=max_tokens, effort=effort, timeout=self.timeout,
            )
        else:
            self.transport = OpenAICompatibleTransport(
                self.model, base_url=base_url or DEFAULT_OPENAI_BASE_URL, api_key_env=api_key_env,
                max_tokens=max_tokens, timeout=self.timeout, http_client=http_client,
                reasoning_effort=reasoning_effort or _env(REASONING_EFFORT_ENV),
                json_schema=json_schema or _env(JSON_SCHEMA_ENV),
                extra_body=extra_body if extra_body is not None else _resolve_extra_body(),
            )

    def recognize(self, timestamp: float, image: np.ndarray) -> list[RecognizedBlock]:
        """Recognize content on a single keyframe (BGR array as OpenCV produces)."""
        png_b64, _ = encode_image_png(image, self.max_image_side)
        reply = self.transport.complete(png_b64)
        return [
            RecognizedBlock(text=block["text"], kind=block["kind"], timestamp=timestamp,
                            confidence=block.get("confidence"))
            for block in parse_blocks(reply, collapse_repeats=self.collapse_repeats)
        ]

    def build_request(self, png_b64: str) -> dict[str, Any]:
        """The provider-specific request for one encoded image (for inspection and adapters)."""
        return self.transport.build_request(png_b64)


def _resolve_extra_body() -> dict[str, Any]:
    raw = _env(EXTRA_BODY_ENV)
    if raw is None:
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        raise ValueError(f"{EXTRA_BODY_ENV} must be a JSON object, got {raw!r}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{EXTRA_BODY_ENV} must be a JSON object, got {raw!r}")
    return data


def _resolve_timeout(timeout: float | None) -> float:
    if timeout is not None:
        return float(timeout)
    raw = _env(TIMEOUT_ENV)
    if raw is None:
        return DEFAULT_TIMEOUT
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{TIMEOUT_ENV} must be a number of seconds, got {raw!r}") from None


# --- command line ---------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Run the recognizer on one image file and print the blocks as JSON."""
    parser = argparse.ArgumentParser(description="Transcribe one board image with the multimodal LLM backend.")
    parser.add_argument("image", help="path to a board image (any format OpenCV can read)")
    parser.add_argument("--provider", choices=PROVIDERS, default=None,
                        help=f"transport to use (default: ${PROVIDER_ENV} or {DEFAULT_PROVIDER})")
    parser.add_argument("--model", default="", help=f"model id (default: ${MODEL_ENV} or the provider default)")
    parser.add_argument("--base-url", default=None, help=f"endpoint base URL (default: ${BASE_URL_ENV})")
    parser.add_argument("--api-key-env", default=None,
                        help=f"name of the environment variable holding the key (default: ${KEY_NAME_ENV})")
    parser.add_argument("--timeout", type=float, default=None, help=f"seconds per request (default: ${TIMEOUT_ENV} or 600)")
    parser.add_argument("--timestamp", type=float, default=0.0, help="timestamp to attach to the blocks")
    parser.add_argument("--max-image-side", type=int, default=MAX_IMAGE_SIDE, help="downscale limit in pixels")
    parser.add_argument("--effort", default=None, help="optional output_config.effort level (anthropic only)")
    parser.add_argument("--reasoning-effort", default=None,
                        help=f"(openai) reasoning_effort to send; auto, omit, or a level (default: ${REASONING_EFFORT_ENV})")
    parser.add_argument("--json-schema", default=None, choices=("auto", "on", "off"),
                        help=f"(openai) send the blocks JSON schema as response_format (default: ${JSON_SCHEMA_ENV} or auto)")
    args = parser.parse_args(argv)

    image = cv2.imread(args.image)
    if image is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 2
    try:
        recognizer = MultimodalLLMRecognizer(
            args.model, provider=args.provider, base_url=args.base_url, api_key_env=args.api_key_env,
            max_image_side=args.max_image_side, effort=args.effort, timeout=args.timeout,
            reasoning_effort=args.reasoning_effort, json_schema=args.json_schema,
        )
        blocks = recognizer.recognize(args.timestamp, image)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps([block.__dict__ for block in blocks], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
