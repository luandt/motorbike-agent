"""Multi-provider LLM layer. Each adapter runs through its real SDK against a mocked HTTP transport,
so request shapes and error mapping are checked without network access or API keys."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from agent import llm as L
from agent.extractor import LLMExtraction
from agent.memory import Store
from agent.models import Message, Sender
from agent.pipeline import Agent


class Out(BaseModel):
    city: str
    score: float


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------- #
# spec parsing and selection
# --------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ["AGENT_LLM", "AGENT_MODEL", "LLM_BASE_URL", "LLM_API_KEY"] + \
             [k for p in L.PROVIDERS.values() for k in p.key_env]:
        monkeypatch.delenv(k, raising=False)


@pytest.mark.parametrize("spec,expected", [
    (None, ("anthropic", "claude-opus-5-5")),
    ("anthropic", ("anthropic", "claude-opus-5-5")),
    ("openai", ("openai", "gpt-6-luna")),
    ("claude-sonnet-5-5", ("anthropic", "claude-sonnet-5-5")),
    ("openai:gpt-test", ("openai", "gpt-test")),
    ("Gemini:gemini-test", ("gemini", "gemini-test")),
    ("ollama:qwen3:8b", ("ollama", "qwen3:8b")),          # model names can contain ':'
])
def test_parse_spec(spec, expected):
    assert L.parse_spec(spec) == expected


def test_parse_spec_from_env(monkeypatch):
    monkeypatch.setenv("AGENT_LLM", "groq:some-model")
    assert L.parse_spec() == ("groq", "some-model")


@pytest.mark.parametrize("spec", ["gemini", "mistral:x"])
def test_parse_spec_errors(spec):
    with pytest.raises(L.LLMUnavailable):
        L.parse_spec(spec)


def test_auto_mode_without_credentials_runs_rules():
    assert L.make_llm("auto", "openai:gpt-test") is None
    assert L.make_llm("auto", "gemini:gemini-test") is None
    assert L.make_llm("rules", "openai:gpt-test") is None


def test_llm_mode_without_credentials_fails_loudly():
    with pytest.raises(L.LLMUnavailable):
        L.make_llm("llm", "openai:gpt-test")


def test_build_openai_compatible_presets(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    x = L.make_llm("auto", "openrouter:vendor/model")
    assert isinstance(x, L.OpenAILLM) and str(x.client.base_url).startswith("https://openrouter.ai/api/v1")
    assert L.make_llm("auto", "ollama:qwen3:8b").name == "ollama:qwen3:8b"     # local, no key needed
    assert L.make_llm("auto", "compatible:m") is None                          # needs LLM_BASE_URL
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    assert str(L.make_llm("auto", "compatible:m").client.base_url).startswith("http://localhost:8000/v1")


# --------------------------------------------------------------------------- #
# OpenAI adapter
# --------------------------------------------------------------------------- #
def chat_completion(content, finish="stop", refusal=None):
    return {"id": "c1", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": content, "refusal": refusal}}]}


def openai_llm(handler):
    import openai
    client = openai.OpenAI(api_key="test", base_url="http://test/v1", max_retries=0, http_client=mock_client(handler))
    return L.OpenAILLM("m", client=client)


def test_openai_strict_schema():
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json=chat_completion(json.dumps({"city": "HCM", "score": 0.9})))

    out = openai_llm(handler).parse("sys", "user", Out)
    assert out == Out(city="HCM", score=0.9)
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["messages"][0] == {"role": "system", "content": "sys"}


def test_openai_compatible_server_without_schema_support_falls_back_to_json_mode():
    calls = []

    def handler(req):
        body = json.loads(req.content)
        calls.append(body["response_format"]["type"])
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, json={"error": {"message": "response_format json_schema not supported"}})
        return httpx.Response(200, json=chat_completion('```json\n{"city": "HN", "score": 1}\n```'))

    x = openai_llm(handler)
    assert x.parse("sys", "user", Out) == Out(city="HN", score=1)
    assert x.parse("sys", "user", Out).city == "HN"
    assert calls == ["json_schema", "json_object", "json_object"]     # remembers JSON mode


@pytest.mark.parametrize("response", [
    httpx.Response(429, json={"error": {"message": "slow down"}}),
    httpx.Response(401, json={"error": {"message": "bad key"}}),
    httpx.Response(500, json={"error": {"message": "boom"}}),
    httpx.Response(200, json=chat_completion(None, refusal="I can't help with that")),
    httpx.Response(200, json=chat_completion('{"city": "HCM"', finish="length")),
    httpx.Response(200, json=chat_completion('{"town": "HCM"}')),
])
def test_openai_failures_become_llm_unavailable(response):
    with pytest.raises(L.LLMUnavailable):
        openai_llm(lambda req: response).parse("sys", "user", Out)


def test_openai_connection_error():
    def handler(req):
        raise httpx.ConnectError("down")
    with pytest.raises(L.LLMUnavailable, match="connection"):
        openai_llm(handler).parse("sys", "user", Out)


# --------------------------------------------------------------------------- #
# Anthropic adapter (unchanged behaviour, now behind the same interface)
# --------------------------------------------------------------------------- #
def anthropic_llm(handler):
    import anthropic
    client = anthropic.Anthropic(api_key="test", base_url="http://test", max_retries=0,
                                 http_client=mock_client(handler))
    return L.AnthropicLLM("claude-opus-5-5", client=client)


def message(text, stop="end_turn"):
    return {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": [{"type": "text", "text": text}], "stop_reason": stop, "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1}}


def test_anthropic_parse():
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json=message(json.dumps({"city": "HCM", "score": 0.5})))

    assert anthropic_llm(handler).parse("sys", "user", Out) == Out(city="HCM", score=0.5)
    assert seen[0]["model"] == "claude-opus-5-5" and seen[0]["fallbacks"] == "default"


@pytest.mark.parametrize("response", [
    httpx.Response(200, json=message("", stop="refusal")),
    httpx.Response(429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "x"}}),
])
def test_anthropic_failures(response):
    with pytest.raises(L.LLMUnavailable):
        anthropic_llm(lambda req: response).parse("sys", "user", Out)


# --------------------------------------------------------------------------- #
# Gemini adapter (fake client: google-genai's transport is not httpx-injectable in a stable way)
# --------------------------------------------------------------------------- #
class FakeModels:
    def __init__(self, text):
        self.text, self.calls = text, []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        return SimpleNamespace(text=self.text, candidates=[SimpleNamespace(finish_reason="SAFETY")])


def test_gemini_parse_sends_schema():
    models = FakeModels(json.dumps({"city": "ĐN", "score": 0.2}))
    x = L.GeminiLLM("gemini-test", client=SimpleNamespace(models=models))
    assert x.parse("sys", "user", Out) == Out(city="ĐN", score=0.2)
    model, contents, config = models.calls[0]
    assert model == "gemini-test" and contents == "user"
    assert config.system_instruction == "sys" and config.response_mime_type == "application/json"
    assert config.response_json_schema["required"] == ["city", "score"]


@pytest.mark.parametrize("text", [None, "not json", '{"city": 1}'])
def test_gemini_bad_output(text):
    x = L.GeminiLLM("gemini-test", client=SimpleNamespace(models=FakeModels(text)))
    with pytest.raises(L.LLMUnavailable):
        x.parse("sys", "user", Out)


# --------------------------------------------------------------------------- #
# The agent runs on any provider object, and still falls back to rules
# --------------------------------------------------------------------------- #
class ScriptedLLM(L.StructuredLLM):
    provider, model = "fake", "scripted"

    def parse(self, system, user, schema):
        self.calls += 1
        if schema is LLMExtraction:
            return LLMExtraction(intents=[], risks=[], facts=[
                {"field": "constraints.vehicle_type", "value": "scooter", "confidence": 0.9, "raw": "xe tay ga"}])
        raise L.LLMUnavailable("planner and responder fall back to rules")


def test_agent_with_custom_provider(tmp_path):
    conv = tmp_path / "conv.json"
    conv.write_text(json.dumps({"z": {"buyer_id": "b", "seller_id": None}}))
    fake = ScriptedLLM()
    agent = Agent(Store(":memory:"), mode="llm", llm=fake, conversations_path=conv)
    assert agent.mode == "llm" and agent.llm_name == "fake:scripted"
    t = agent.handle(Message(conversation_id="z", msg_id="z#1", index=1, sender=Sender.BUYER,
                             text="Mình cần xe đi làm tầm 25 củ ở Sài Gòn",
                             timestamp=datetime(2026, 3, 1, tzinfo=timezone.utc)))
    assert t.state.cval("vehicle_type") == "scooter"        # the LLM added a fact the rules missed
    assert t.state.cval("budget_max") == 25_000_000          # rules still own the numbers
    assert t.actions and fake.calls >= 2                     # planner fell back to rules, turn completed
