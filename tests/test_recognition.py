"""Unit tests for the multimodal LLM recognition backend.

No network: a fake SDK client stands in for ``anthropic.Anthropic`` and a fake HTTP client for the
OpenAI-compatible transport. The two ``real_llm`` tests at the end are skipped unless configured.
"""

import base64
import http.server
import io
import json
import os
import threading
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from whiteboard_extraction.recognition.base import RecognizedBlock
from whiteboard_extraction.recognition.llm_multimodal import (
    API_KEY_ENV,
    BASE_URL_ENV,
    DEFAULT_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TIMEOUT,
    EXTRA_BODY_ENV,
    JSON_SCHEMA_ENV,
    KEY_NAME_ENV,
    MAX_IMAGE_SIDE,
    MODEL_ENV,
    PROVIDER_ENV,
    REASONING_EFFORT_ENV,
    RESPONSE_FORMAT,
    SYSTEM_PROMPT,
    TIMEOUT_ENV,
    USER_PROMPT,
    MultimodalLLMRecognizer,
    UrllibHTTPClient,
    encode_image_png,
    parse_blocks,
    split_think,
)

WCE_ENV_VARS = (PROVIDER_ENV, MODEL_ENV, BASE_URL_ENV, KEY_NAME_ENV, TIMEOUT_ENV, REASONING_EFFORT_ENV,
                JSON_SCHEMA_ENV, EXTRA_BODY_ENV)


@pytest.fixture(autouse=True)
def isolate_env(request, monkeypatch):
    """Keep the developer's shell configuration out of the unit tests (the live tests keep it)."""
    if "real_llm" in request.keywords:
        return
    for name in WCE_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


class FakeClient:
    """Stands in for ``anthropic.Anthropic``: records requests and returns a canned reply."""

    def __init__(self, reply="[]", *, stop_reason="end_turn", error=None, content=None, stop_details=None):
        self.reply = reply
        self.stop_reason = stop_reason
        self.error = error
        self.content = content
        self.stop_details = stop_details
        self.requests = []
        self.messages = self  # so that client.messages.create(...) resolves to self.create

    def create(self, **request):
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        content = self.content if self.content is not None else [SimpleNamespace(type="text", text=self.reply)]
        return SimpleNamespace(content=content, stop_reason=self.stop_reason, stop_details=self.stop_details)


class FakeHTTP:
    """Stands in for ``UrllibHTTPClient``: records POSTs and returns a canned chat-completion body."""

    def __init__(self, reply="[]", *, data=None, error=None, finish_reason="stop"):
        self.reply = reply
        self.data = data
        self.error = error
        self.finish_reason = finish_reason
        self.calls = []

    def post(self, url, headers, payload, timeout):
        self.calls.append(SimpleNamespace(url=url, headers=headers, payload=payload, timeout=timeout))
        if self.error is not None:
            raise self.error
        if self.data is not None:
            return self.data
        message = {"role": "assistant", "content": self.reply}
        return {"choices": [{"index": 0, "message": message, "finish_reason": self.finish_reason}]}


def openai_recognizer(http, model="", **kwargs):
    return MultimodalLLMRecognizer(model, provider="openai", http_client=http, **kwargs)


def board(width=640, height=360, color=(255, 255, 255)):
    """A plain BGR board image of the given size."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def synthetic_board():
    image = board(width=1200, height=600)
    cv2.putText(image, "Pythagorean theorem", (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4)
    cv2.putText(image, "a^2 + b^2 = c^2", (60, 350), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 0), 5)
    return image


def decode_png(data_b64):
    return Image.open(io.BytesIO(base64.b64decode(data_b64)))


def assert_board_blocks(blocks, timestamp):
    assert blocks, "expected at least one block for a board with writing on it"
    assert {b.kind for b in blocks} <= {"prose", "equation"}
    assert all(b.timestamp == timestamp for b in blocks)
    joined = " ".join(b.text for b in blocks).lower()
    assert "pythagorean" in joined or "c^2" in joined.replace(" ", "")


# --- constructor and provider selection -----------------------------------------------------------

def test_empty_model_name_selects_default_model():
    assert MultimodalLLMRecognizer("").model == DEFAULT_MODEL
    assert MultimodalLLMRecognizer("  ").model == DEFAULT_MODEL
    assert MultimodalLLMRecognizer("claude-sonnet-5-5").model == "claude-sonnet-5-5"


def test_model_is_first_positional_argument():
    # The registry and web app construct the backend as MultimodalLLMRecognizer(name).
    assert MultimodalLLMRecognizer("some-model").model == "some-model"
    assert MultimodalLLMRecognizer("some-model", provider="openai", http_client=FakeHTTP()).model == "some-model"


def test_provider_defaults_to_anthropic():
    recognizer = MultimodalLLMRecognizer("")
    assert recognizer.provider == "anthropic"
    assert recognizer.timeout == DEFAULT_TIMEOUT
    assert type(recognizer.transport).__name__ == "AnthropicTransport"


def test_openai_provider_has_ollama_defaults():
    recognizer = openai_recognizer(FakeHTTP())
    assert recognizer.provider == "openai"
    assert recognizer.model == DEFAULT_OPENAI_MODEL
    assert recognizer.transport.url == DEFAULT_OPENAI_BASE_URL + "/chat/completions"
    assert recognizer.transport.api_key_env is None


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="unknown multimodal LLM provider 'gemini'"):
        MultimodalLLMRecognizer("", provider="gemini")


def test_env_selects_openai_provider_and_settings(monkeypatch):
    monkeypatch.setenv(PROVIDER_ENV, "openai")
    monkeypatch.setenv(MODEL_ENV, "some-vision-model")
    monkeypatch.setenv(BASE_URL_ENV, "https://example.test/api/v1/")
    monkeypatch.setenv(KEY_NAME_ENV, "EXAMPLE_KEY")
    monkeypatch.setenv("EXAMPLE_KEY", "secret-123")
    monkeypatch.setenv(TIMEOUT_ENV, "45")
    http = FakeHTTP("[]")

    recognizer = MultimodalLLMRecognizer("", http_client=http)
    recognizer.recognize(0.0, board())

    assert recognizer.provider == "openai"
    assert recognizer.model == "some-vision-model"
    call = http.calls[0]
    assert call.url == "https://example.test/api/v1/chat/completions"
    assert call.headers["Authorization"] == "Bearer secret-123"
    assert call.timeout == 45.0
    assert call.payload["model"] == "some-vision-model"


def test_constructor_arguments_beat_environment(monkeypatch):
    monkeypatch.setenv(PROVIDER_ENV, "anthropic")
    monkeypatch.setenv(MODEL_ENV, "env-model")
    monkeypatch.setenv(BASE_URL_ENV, "https://env.test/v1")
    monkeypatch.setenv(TIMEOUT_ENV, "10")
    http = FakeHTTP("[]")

    recognizer = openai_recognizer(http, model="arg-model", base_url="https://arg.test/v1", timeout=3)
    recognizer.recognize(0.0, board())

    assert recognizer.model == "arg-model"
    assert http.calls[0].url == "https://arg.test/v1/chat/completions"
    assert http.calls[0].timeout == 3.0


def test_bad_timeout_env_is_rejected(monkeypatch):
    monkeypatch.setenv(TIMEOUT_ENV, "soon")
    with pytest.raises(ValueError, match=TIMEOUT_ENV):
        MultimodalLLMRecognizer("")


# --- anthropic provider: successful recognition ---------------------------------------------------

def test_recognize_parses_json_reply_into_blocks():
    reply = json.dumps([
        {"text": "Pythagorean theorem", "kind": "prose"},
        {"text": "a^2 + b^2 = c^2", "kind": "equation"},
    ])
    client = FakeClient(reply)
    blocks = MultimodalLLMRecognizer("", client=client).recognize(12.5, board())

    assert blocks == [
        RecognizedBlock(text="Pythagorean theorem", kind="prose", timestamp=12.5),
        RecognizedBlock(text="a^2 + b^2 = c^2", kind="equation", timestamp=12.5),
    ]
    assert all(block.confidence is None for block in blocks)


def test_recognize_strips_markdown_code_fences():
    reply = 'Here you go:\n```json\n[{"text": "x = 1", "kind": "equation"}]\n```\n'
    blocks = MultimodalLLMRecognizer("", client=FakeClient(reply)).recognize(1.0, board())
    assert [(b.text, b.kind) for b in blocks] == [("x = 1", "equation")]


def test_recognize_blank_board_returns_empty_list():
    assert MultimodalLLMRecognizer("", client=FakeClient("[]")).recognize(0.0, board()) == []


def test_recognize_drops_blocks_with_empty_text():
    reply = json.dumps([
        {"text": "   ", "kind": "prose"},
        {"text": "", "kind": "equation"},
        {"text": "kept", "kind": "prose"},
    ])
    blocks = MultimodalLLMRecognizer("", client=FakeClient(reply)).recognize(0.0, board())
    assert [b.text for b in blocks] == ["kept"]


def test_consecutive_repeated_blocks_are_collapsed():
    formula = {"text": "m = \\frac{y_2-y_1}{x_2-x_1}", "kind": "equation"}
    reply = json.dumps([formula, formula, {"text": "m =  \\frac{y_2-y_1}{x_2-x_1}", "kind": "equation"},
                        {"text": "Equations:", "kind": "prose"}, formula, formula])
    blocks = MultimodalLLMRecognizer("", client=FakeClient(reply)).recognize(0.0, board())
    assert [b.text for b in blocks] == ["m = \\frac{y_2-y_1}{x_2-x_1}", "Equations:", "m = \\frac{y_2-y_1}{x_2-x_1}"]

    same_text_other_kind = [{"text": "x = 1", "kind": "equation"}, {"text": "x = 1", "kind": "prose"}]
    assert len(parse_blocks(json.dumps(same_text_other_kind))) == 2  # a kind change is not a repeat
    assert len(parse_blocks(json.dumps([formula] * 5), collapse_repeats=False)) == 5


def test_collapse_can_be_disabled_on_the_recognizer():
    reply = json.dumps([{"text": "a", "kind": "prose"}] * 3)
    assert len(MultimodalLLMRecognizer("", client=FakeClient(reply), collapse_repeats=False).recognize(0.0, board())) == 3


def test_recognize_keeps_numeric_confidence_when_present():
    reply = json.dumps([{"text": "E = mc^2", "kind": "equation", "confidence": 0.9}])
    blocks = MultimodalLLMRecognizer("", client=FakeClient(reply)).recognize(0.0, board())
    assert blocks[0].confidence == pytest.approx(0.9)


def test_recognize_concatenates_multiple_text_blocks():
    content = [
        SimpleNamespace(type="thinking", thinking="..."),
        SimpleNamespace(type="text", text='[{"text": "part one", '),
        SimpleNamespace(type="text", text='"kind": "prose"}]'),
    ]
    blocks = MultimodalLLMRecognizer("", client=FakeClient(content=content)).recognize(0.0, board())
    assert [b.text for b in blocks] == ["part one"]


# --- anthropic provider: request shape ------------------------------------------------------------

def test_request_sends_png_image_before_prompt_and_uses_model():
    client = FakeClient("[]")
    image = board(width=64, height=32, color=(255, 0, 0))  # pure blue in BGR
    MultimodalLLMRecognizer("my-model", client=client).recognize(0.0, image)

    assert len(client.requests) == 1
    request = client.requests[0]
    assert request["model"] == "my-model"
    assert request["max_tokens"] > 0
    assert "JSON array" in request["system"]
    assert "exactly once" in request["system"] and "equals sign" in request["system"]
    assert "output_config" not in request

    (message,) = request["messages"]
    assert message["role"] == "user"
    image_block, text_block = message["content"]
    assert image_block["type"] == "image"
    assert image_block["source"]["type"] == "base64"
    assert image_block["source"]["media_type"] == "image/png"
    assert text_block == {"type": "text", "text": USER_PROMPT}

    png = decode_png(image_block["source"]["data"])
    assert png.format == "PNG"
    assert png.size == (64, 32)
    assert png.convert("RGB").getpixel((0, 0)) == (0, 0, 255)  # BGR blue became RGB blue


def test_effort_is_passed_through_output_config():
    client = FakeClient("[]")
    MultimodalLLMRecognizer("", client=client, effort="low").recognize(0.0, board())
    assert client.requests[0]["output_config"] == {"effort": "low"}


# --- image encoding -------------------------------------------------------------------------------

def test_encode_image_downscales_when_a_side_exceeds_the_limit():
    data, (width, height) = encode_image_png(board(width=4000, height=1000))
    assert (width, height) == (MAX_IMAGE_SIDE, 644)
    assert decode_png(data).size == (MAX_IMAGE_SIDE, 644)

    data, (width, height) = encode_image_png(board(width=500, height=3000), max_side=1500)
    assert (width, height) == (250, 1500)


def test_encode_image_leaves_small_images_alone():
    data, size = encode_image_png(board(width=800, height=600))
    assert size == (800, 600)
    assert decode_png(data).size == (800, 600)


def test_encode_image_accepts_grayscale_and_bgra():
    gray = np.full((20, 30), 128, dtype=np.uint8)
    data, size = encode_image_png(gray)
    assert size == (30, 20)
    assert decode_png(data).mode == "RGB"

    bgra = np.zeros((20, 30, 4), dtype=np.uint8)
    bgra[..., 2] = 255  # red channel in BGRA
    bgra[..., 3] = 255
    data, _ = encode_image_png(bgra)
    assert decode_png(data).convert("RGB").getpixel((0, 0)) == (255, 0, 0)


def test_encode_image_rejects_bad_input():
    with pytest.raises(ValueError):
        encode_image_png(np.zeros((0, 10, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_image_png(np.zeros((10, 10, 2), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_image_png("not an array")


def test_recognize_downscales_the_sent_image():
    client = FakeClient("[]")
    MultimodalLLMRecognizer("", client=client, max_image_side=800).recognize(0.0, board(width=1600, height=400))
    sent = client.requests[0]["messages"][0]["content"][0]["source"]["data"]
    assert decode_png(sent).size == (800, 200)


# --- anthropic provider: failures -----------------------------------------------------------------

def test_invalid_kind_is_rejected():
    reply = json.dumps([{"text": "hello", "kind": "diagram"}])
    with pytest.raises(RuntimeError, match="invalid kind 'diagram'"):
        MultimodalLLMRecognizer("", client=FakeClient(reply)).recognize(0.0, board())


def test_missing_text_field_is_rejected():
    reply = json.dumps([{"kind": "prose"}])
    with pytest.raises(RuntimeError, match="no string 'text'"):
        MultimodalLLMRecognizer("", client=FakeClient(reply)).recognize(0.0, board())


def test_unparseable_reply_raises_runtime_error_with_excerpt():
    reply = "I'm sorry, I cannot read this board."
    with pytest.raises(RuntimeError) as info:
        MultimodalLLMRecognizer("", client=FakeClient(reply)).recognize(0.0, board())
    assert "not valid JSON" in str(info.value)
    assert reply in str(info.value)


def test_non_array_json_is_rejected():
    with pytest.raises(RuntimeError, match="not a JSON array"):
        MultimodalLLMRecognizer("", client=FakeClient('{"text": "x", "kind": "prose"}')).recognize(0.0, board())


def test_api_error_becomes_runtime_error():
    class BoomError(Exception):
        pass

    client = FakeClient(error=BoomError("rate limited"))
    with pytest.raises(RuntimeError, match=r"request failed \(BoomError\): rate limited") as info:
        MultimodalLLMRecognizer("", client=client).recognize(0.0, board())
    assert isinstance(info.value.__cause__, BoomError)


def test_refusal_stop_reason_raises():
    client = FakeClient("", stop_reason="refusal", stop_details=SimpleNamespace(explanation="policy"))
    with pytest.raises(RuntimeError, match="refused the request: policy"):
        MultimodalLLMRecognizer("", client=client).recognize(0.0, board())


def test_truncated_reply_raises():
    client = FakeClient('[{"text": "a", "kind": "pro', stop_reason="max_tokens")
    with pytest.raises(RuntimeError, match="max_tokens"):
        MultimodalLLMRecognizer("", client=client).recognize(0.0, board())


def test_empty_response_raises():
    client = FakeClient(content=[])
    with pytest.raises(RuntimeError, match="no text"):
        MultimodalLLMRecognizer("", client=client).recognize(0.0, board())


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=API_KEY_ENV):
        MultimodalLLMRecognizer("").recognize(0.0, board())


def test_api_key_is_read_at_call_time(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    recognizer = MultimodalLLMRecognizer("")  # constructing without a key is fine
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    client = recognizer.transport._get_client()
    assert type(client).__name__ == "Anthropic"


def test_real_sdk_client_errors_are_wrapped():
    # Exercises the real SDK request path (kwargs accepted, error chained) against a closed local port.
    anthropic = pytest.importorskip("anthropic")
    client = anthropic.Anthropic(api_key="test-key", base_url="http://127.0.0.1:9", max_retries=0, timeout=2.0)
    with pytest.raises(RuntimeError, match="request failed") as info:
        MultimodalLLMRecognizer("", client=client).recognize(0.0, board())
    assert isinstance(info.value.__cause__, anthropic.APIConnectionError)


# --- openai-compatible provider: request shape ----------------------------------------------------

def test_openai_request_shape_without_key():
    http = FakeHTTP("[]")
    image = board(width=64, height=32, color=(255, 0, 0))  # pure blue in BGR
    openai_recognizer(http, timeout=120).recognize(0.0, image)

    assert len(http.calls) == 1
    call = http.calls[0]
    assert call.url == "http://localhost:11434/v1/chat/completions"
    assert call.headers == {"Content-Type": "application/json", "Accept": "application/json"}
    assert call.timeout == 120.0

    payload = call.payload
    assert payload["model"] == DEFAULT_OPENAI_MODEL
    assert payload["max_tokens"] > 0
    assert payload["reasoning_effort"] == "low"  # local server: keep thinking models from eating the context
    assert payload["response_format"] == RESPONSE_FORMAT  # local server: pin the reply shape
    assert payload["response_format"]["json_schema"]["schema"]["items"]["properties"]["kind"]["enum"] == ["prose", "equation"]
    system, user = payload["messages"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert user["role"] == "user"
    image_part, text_part = user["content"]
    assert image_part["type"] == "image_url"
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
    assert text_part == {"type": "text", "text": USER_PROMPT}

    png = decode_png(image_part["image_url"]["url"].split(",", 1)[1])
    assert png.format == "PNG"
    assert png.size == (64, 32)
    assert png.convert("RGB").getpixel((0, 0)) == (0, 0, 255)


def test_openai_request_sends_bearer_key_and_normalises_base_url(monkeypatch):
    monkeypatch.setenv("MY_PROVIDER_KEY", "sk-test")
    http = FakeHTTP("[]")
    recognizer = openai_recognizer(http, model="vendor/vision-model", base_url="https://api.example.test/v1/",
                                   api_key_env="MY_PROVIDER_KEY")
    recognizer.recognize(0.0, board())

    call = http.calls[0]
    assert call.url == "https://api.example.test/v1/chat/completions"
    assert call.headers["Authorization"] == "Bearer sk-test"
    assert call.payload["model"] == "vendor/vision-model"


def test_openai_missing_key_gives_clear_error_before_any_request(monkeypatch):
    monkeypatch.delenv("MY_PROVIDER_KEY", raising=False)
    http = FakeHTTP("[]")
    with pytest.raises(RuntimeError, match="MY_PROVIDER_KEY"):
        openai_recognizer(http, api_key_env="MY_PROVIDER_KEY").recognize(0.0, board())
    assert http.calls == []


def test_openai_downscales_the_sent_image():
    http = FakeHTTP("[]")
    openai_recognizer(http, max_image_side=640).recognize(0.0, board(width=1280, height=720))
    url = http.calls[0].payload["messages"][1]["content"][0]["image_url"]["url"]
    assert decode_png(url.split(",", 1)[1]).size == (640, 360)


# --- openai-compatible provider: parsing and failures ----------------------------------------------

def test_openai_parses_reply_with_prose_and_fences():
    reply = 'Sure, here is the board:\n```json\n[{"text": "Chain rule", "kind": "prose"}, ' \
            '{"text": "(f \\\\circ g)\' = f\'(g) g\'", "kind": "equation"}]\n```\nHope this helps!'
    blocks = openai_recognizer(FakeHTTP(reply)).recognize(7.0, board())
    assert blocks == [
        RecognizedBlock(text="Chain rule", kind="prose", timestamp=7.0),
        RecognizedBlock(text="(f \\circ g)' = f'(g) g'", kind="equation", timestamp=7.0),
    ]


def test_openai_blank_board_and_invalid_kind():
    assert openai_recognizer(FakeHTTP("[]")).recognize(0.0, board()) == []
    with pytest.raises(RuntimeError, match="invalid kind 'figure'"):
        openai_recognizer(FakeHTTP('[{"text": "x", "kind": "figure"}]')).recognize(0.0, board())


def test_openai_content_may_be_a_list_of_parts():
    data = {"choices": [{"message": {"content": [
        {"type": "text", "text": '[{"text": "y = x", '},
        {"type": "text", "text": '"kind": "equation"}]'},
    ]}, "finish_reason": "stop"}]}
    blocks = openai_recognizer(FakeHTTP(data=data)).recognize(0.0, board())
    assert [b.text for b in blocks] == ["y = x"]


def test_openai_http_error_keeps_the_server_message():
    # The body is what Ollama 0.15 really returns for a model that has not been pulled.
    error = RuntimeError("HTTP 404 from http://localhost:11434/v1/chat/completions: "
                         '{"error":{"message":"model \'qwen3-vl:4b\' not found","type":"api_error"}}')
    with pytest.raises(RuntimeError, match="request failed: HTTP 404 .*model 'qwen3-vl:4b' not found") as info:
        openai_recognizer(FakeHTTP(error=error)).recognize(0.0, board())
    assert info.value.__cause__ is error
    assert f"pull it first: ollama pull {DEFAULT_OPENAI_MODEL}" in str(info.value)


def test_openai_pull_hint_only_for_local_servers():
    error = RuntimeError("HTTP 404 from https://api.example.test/v1/chat/completions: model 'x' not found")
    with pytest.raises(RuntimeError) as info:
        openai_recognizer(FakeHTTP(error=error), base_url="https://api.example.test/v1").recognize(0.0, board())
    assert "ollama pull" not in str(info.value)


def test_openai_other_exceptions_are_wrapped():
    with pytest.raises(RuntimeError, match=r"request failed \(ConnectionError\): boom"):
        openai_recognizer(FakeHTTP(error=ConnectionError("boom"))).recognize(0.0, board())


def test_openai_truncated_reply_raises():
    with pytest.raises(RuntimeError, match="max_tokens"):
        openai_recognizer(FakeHTTP('[{"text": "a', finish_reason="length")).recognize(0.0, board())


def test_openai_content_filter_raises():
    with pytest.raises(RuntimeError, match="refused"):
        openai_recognizer(FakeHTTP("", finish_reason="content_filter")).recognize(0.0, board())


def test_openai_error_payload_and_missing_choices_raise():
    with pytest.raises(RuntimeError, match="API error: .*quota"):
        openai_recognizer(FakeHTTP(data={"error": {"message": "quota exceeded"}})).recognize(0.0, board())
    with pytest.raises(RuntimeError, match="no choices"):
        openai_recognizer(FakeHTTP(data={"choices": []})).recognize(0.0, board())
    with pytest.raises(RuntimeError, match="no text"):
        openai_recognizer(FakeHTTP(data={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]})) \
            .recognize(0.0, board())


# --- openai-compatible provider: thinking models --------------------------------------------------

def test_openai_reasoning_effort_defaults_depend_on_the_host(monkeypatch):
    monkeypatch.setenv("HOSTED_KEY", "k")
    local, hosted = FakeHTTP("[]"), FakeHTTP("[]")
    openai_recognizer(local).recognize(0.0, board())
    openai_recognizer(hosted, base_url="https://api.example.test/v1", api_key_env="HOSTED_KEY").recognize(0.0, board())
    assert local.calls[0].payload["reasoning_effort"] == "low"
    assert "reasoning_effort" not in hosted.calls[0].payload
    assert "response_format" in local.calls[0].payload
    assert "response_format" not in hosted.calls[0].payload


def test_openai_reasoning_effort_can_be_set_or_omitted(monkeypatch):
    explicit, omitted, from_env = FakeHTTP("[]"), FakeHTTP("[]"), FakeHTTP("[]")
    openai_recognizer(explicit, reasoning_effort="none").recognize(0.0, board())
    openai_recognizer(omitted, reasoning_effort="omit").recognize(0.0, board())
    monkeypatch.setenv(REASONING_EFFORT_ENV, "medium")
    openai_recognizer(from_env).recognize(0.0, board())
    assert explicit.calls[0].payload["reasoning_effort"] == "none"
    assert "reasoning_effort" not in omitted.calls[0].payload
    assert from_env.calls[0].payload["reasoning_effort"] == "medium"


def test_openai_json_schema_can_be_forced_on_or_off(monkeypatch):
    monkeypatch.setenv("HOSTED_KEY", "k")
    on, off, from_env = FakeHTTP("[]"), FakeHTTP("[]"), FakeHTTP("[]")
    openai_recognizer(on, base_url="https://api.example.test/v1", api_key_env="HOSTED_KEY", json_schema="on") \
        .recognize(0.0, board())
    openai_recognizer(off, json_schema="off").recognize(0.0, board())
    monkeypatch.setenv(JSON_SCHEMA_ENV, "off")
    openai_recognizer(from_env).recognize(0.0, board())
    assert on.calls[0].payload["response_format"] == RESPONSE_FORMAT
    assert "response_format" not in off.calls[0].payload
    assert "response_format" not in from_env.calls[0].payload
    with pytest.raises(ValueError, match=JSON_SCHEMA_ENV):
        openai_recognizer(FakeHTTP(), json_schema="maybe")


def test_openai_extra_body_is_merged_last(monkeypatch):
    http = FakeHTTP("[]")
    openai_recognizer(http, extra_body={"temperature": 0, "reasoning_effort": "high"}).recognize(0.0, board())
    assert http.calls[0].payload["temperature"] == 0
    assert http.calls[0].payload["reasoning_effort"] == "high"

    monkeypatch.setenv(EXTRA_BODY_ENV, '{"think": false}')
    http = FakeHTTP("[]")
    openai_recognizer(http).recognize(0.0, board())
    assert http.calls[0].payload["think"] is False

    monkeypatch.setenv(EXTRA_BODY_ENV, "[1, 2]")
    with pytest.raises(ValueError, match=EXTRA_BODY_ENV):
        openai_recognizer(FakeHTTP())
    monkeypatch.setenv(EXTRA_BODY_ENV, "not json")
    with pytest.raises(ValueError, match=EXTRA_BODY_ENV):
        openai_recognizer(FakeHTTP())


class QueuedHTTP(FakeHTTP):
    """A FakeHTTP whose successive POSTs raise or return the queued items in order."""

    def __init__(self, *outcomes):
        super().__init__()
        self.outcomes = list(outcomes)

    def post(self, url, headers, payload, timeout):
        self.calls.append(SimpleNamespace(url=url, headers=headers, payload=payload, timeout=timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return {"choices": [{"message": {"content": outcome}, "finish_reason": "stop"}]}


NO_THINKING = RuntimeError('HTTP 400 from http://localhost:11434/v1/chat/completions: {"error":{"message":'
                           '"\\"llava:latest\\" does not support thinking","type":"invalid_request_error"}}')


def test_openai_retries_without_reasoning_effort_when_the_model_rejects_it():
    # Real Ollama behaviour: a non-thinking model answers HTTP 400 to reasoning_effort.
    http = QueuedHTTP(NO_THINKING, '[{"text": "ok", "kind": "prose"}]', '[]')
    recognizer = openai_recognizer(http, model="llava:latest")

    assert [b.text for b in recognizer.recognize(0.0, board())] == ["ok"]
    assert len(http.calls) == 2
    assert http.calls[0].payload["reasoning_effort"] == "low"
    assert "reasoning_effort" not in http.calls[1].payload

    recognizer.recognize(1.0, board())  # the next frame does not repeat the rejected field
    assert len(http.calls) == 3
    assert "reasoning_effort" not in http.calls[2].payload


NO_SCHEMA = RuntimeError('HTTP 400 from http://localhost:11434/v1/chat/completions: '
                         '{"error":{"message":"response_format json_schema is not supported"}}')


def test_openai_retries_without_response_format_when_the_server_rejects_it():
    http = QueuedHTTP(NO_SCHEMA, '[{"text": "ok", "kind": "prose"}]', '[]')
    recognizer = openai_recognizer(http)
    assert [b.text for b in recognizer.recognize(0.0, board())] == ["ok"]
    assert "response_format" in http.calls[0].payload and "reasoning_effort" in http.calls[0].payload
    assert "response_format" not in http.calls[1].payload and http.calls[1].payload["reasoning_effort"] == "low"
    recognizer.recognize(1.0, board())
    assert "response_format" not in http.calls[2].payload


def test_openai_explicit_json_schema_is_not_retried():
    http = QueuedHTTP(NO_SCHEMA, "[]")
    with pytest.raises(RuntimeError, match="not supported"):
        openai_recognizer(http, json_schema="on").recognize(0.0, board())
    assert len(http.calls) == 1


def test_openai_explicit_reasoning_effort_is_not_retried():
    http = QueuedHTTP(NO_THINKING, "[]")
    with pytest.raises(RuntimeError, match="does not support thinking"):
        openai_recognizer(http, reasoning_effort="low").recognize(0.0, board())
    assert len(http.calls) == 1


def test_openai_other_client_errors_are_not_retried():
    http = QueuedHTTP(RuntimeError("HTTP 401 from https://x/v1/chat/completions: invalid api key"), "[]")
    with pytest.raises(RuntimeError, match="HTTP 401"):
        openai_recognizer(http).recognize(0.0, board())
    assert len(http.calls) == 1


def test_openai_reasoning_only_reply_is_reported_clearly():
    # What Ollama returns when a thinking model fills the context window before answering.
    data = {"choices": [{"message": {"role": "assistant", "content": "", "reasoning": "Okay, let me look at the board..."},
                         "finish_reason": "length"}],
            "usage": {"completion_tokens": 2746, "total_tokens": 4096}}
    with pytest.raises(RuntimeError) as info:
        openai_recognizer(FakeHTTP(data=data)).recognize(0.0, board())
    message = str(info.value)
    assert "spent its reply on reasoning" in message
    assert "finish_reason='length'" in message
    assert REASONING_EFFORT_ENV in message
    assert "Okay, let me look" in message
    assert "Partial reply: ''" not in message


def test_openai_reasoning_content_field_is_recognised_too():
    data = {"choices": [{"message": {"content": None, "reasoning_content": "thinking..."}, "finish_reason": "stop"}]}
    with pytest.raises(RuntimeError, match="spent its reply on reasoning"):
        openai_recognizer(FakeHTTP(data=data)).recognize(0.0, board())


def test_openai_inline_think_tags_are_stripped():
    reply = '<think>\nThe board shows a title and a formula.\n</think>\n[{"text": "Chain rule", "kind": "prose"}]'
    blocks = openai_recognizer(FakeHTTP(reply)).recognize(4.0, board())
    assert blocks == [RecognizedBlock(text="Chain rule", kind="prose", timestamp=4.0)]


def test_openai_unterminated_think_tag_counts_as_reasoning_only():
    with pytest.raises(RuntimeError, match="spent its reply on reasoning"):
        openai_recognizer(FakeHTTP("<think>still thinking about the", finish_reason="length")).recognize(0.0, board())


def test_split_think_and_parser_handle_think_tags():
    assert split_think("<think>a</think>[1]<THINK>b</THINK>") == ("a\nb", "[1]")
    assert split_think("no tags") == ("", "no tags")
    assert parse_blocks('<think>hmm</think>```json\n[{"text": "x", "kind": "prose"}]\n```') == [{"text": "x", "kind": "prose"}]


# --- openai-compatible provider: the real urllib client against a loopback server -------------------

class _ChatCompletionsHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.requests.append(SimpleNamespace(path=self.path, headers=dict(self.headers), body=body))
        status, reply = self.server.responses.pop(0)
        payload = json.dumps(reply).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep pytest output quiet
        pass


@pytest.fixture
def loopback_server():
    server = http.server.HTTPServer(("127.0.0.1", 0), _ChatCompletionsHandler)
    server.requests = []
    server.responses = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


def test_urllib_client_round_trip_against_loopback_server(loopback_server, monkeypatch):
    monkeypatch.setenv("LOOP_KEY", "loop-secret")
    base_url = f"http://127.0.0.1:{loopback_server.server_address[1]}/v1"
    reply = json.dumps([{"text": "Newton's second law", "kind": "prose"}, {"text": "F = ma", "kind": "equation"}])
    loopback_server.responses.append((200, {"choices": [{"message": {"content": reply}, "finish_reason": "stop"}]}))

    recognizer = MultimodalLLMRecognizer("local-model", provider="openai", base_url=base_url,
                                         api_key_env="LOOP_KEY", timeout=5)
    blocks = recognizer.recognize(2.0, board(width=100, height=50))

    assert [(b.text, b.kind) for b in blocks] == [("Newton's second law", "prose"), ("F = ma", "equation")]
    (request,) = loopback_server.requests
    assert request.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer loop-secret"
    assert request.headers["Content-Type"] == "application/json"
    assert request.body["model"] == "local-model"
    assert request.body["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_urllib_client_reports_http_errors_with_server_message(loopback_server):
    base_url = f"http://127.0.0.1:{loopback_server.server_address[1]}/v1"
    loopback_server.responses.append((404, {"error": {"message": "model 'nope' not found"}}))

    with pytest.raises(RuntimeError, match="HTTP 404 .*model 'nope' not found.*ollama pull nope"):
        MultimodalLLMRecognizer("nope", provider="openai", base_url=base_url, timeout=5).recognize(0.0, board())


def test_urllib_client_connection_refused_hints_at_ollama():
    with pytest.raises(RuntimeError, match=r"could not reach .*ollama serve"):
        MultimodalLLMRecognizer("", provider="openai", base_url="http://127.0.0.1:9/v1", timeout=2) \
            .recognize(0.0, board())


def test_urllib_client_rejects_non_json_reply(loopback_server):
    class RawHandler(_ChatCompletionsHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))  # drain the body before answering
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html>not json</html>")

    loopback_server.RequestHandlerClass = RawHandler
    url = f"http://127.0.0.1:{loopback_server.server_address[1]}/v1/chat/completions"
    with pytest.raises(RuntimeError, match="non-JSON reply"):
        UrllibHTTPClient().post(url, {"Content-Type": "application/json"}, {"model": "x"}, 5)


# --- parser unit tests ----------------------------------------------------------------------------

def test_parse_blocks_handles_prose_around_the_array():
    reply = 'Sure! [{"text": "y = mx + b", "kind": "Equation"}] Let me know if you need more.'
    assert parse_blocks(reply) == [{"text": "y = mx + b", "kind": "equation"}]


def test_parse_blocks_accepts_blocks_wrapper_object():
    assert parse_blocks('{"blocks": [{"text": "hi", "kind": "prose"}]}') == [{"text": "hi", "kind": "prose"}]


# --- optional live tests ---------------------------------------------------------------------------

@pytest.mark.real_llm
@pytest.mark.skipif(not os.environ.get(API_KEY_ENV), reason=f"{API_KEY_ENV} not set")
def test_real_anthropic_api_reads_a_synthetic_board():
    blocks = MultimodalLLMRecognizer("", provider="anthropic").recognize(3.0, synthetic_board())
    assert_board_blocks(blocks, 3.0)


@pytest.mark.real_llm
@pytest.mark.skipif(os.environ.get(PROVIDER_ENV, "").strip().lower() != "openai",
                    reason=f"{PROVIDER_ENV} is not 'openai' (set it to test Ollama or a hosted endpoint)")
def test_real_openai_compatible_endpoint_reads_a_synthetic_board():
    # Everything else comes from the WCE_LLM_* variables; the board is kept small for local models' context windows.
    blocks = MultimodalLLMRecognizer("", max_image_side=640).recognize(3.0, synthetic_board())
    assert_board_blocks(blocks, 3.0)
