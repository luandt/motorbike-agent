"""Structured (Pydantic) LLM calls behind one interface, for several providers.

Every LLM call in the agent goes through `parse(system, user, Schema) -> Schema`.
Any failure (no credentials, network, rate limit, refusal, invalid output)
raises `LLMUnavailable`, and the caller falls back to the rule-based component.
That keeps the agent running and makes the fallback rate a measurable metric.

Choose a model with a spec string "provider:model":

  anthropic:claude-opus-5-5      Anthropic SDK, messages.parse + server-side refusal fallback (default)
  openai:gpt-6-luna              OpenAI SDK, chat.completions.parse (strict JSON schema); default model gpt-6-luna
  gemini:<model>                 google-genai SDK, JSON output constrained by the schema
  ollama:<model>                 local Ollama through its OpenAI-compatible endpoint
  openrouter:<model>             OpenRouter        (OPENROUTER_API_KEY)
  deepseek:<model>               DeepSeek          (DEEPSEEK_API_KEY)
  groq:<model>                   Groq              (GROQ_API_KEY)
  compatible:<model>             any other OpenAI-compatible server (LLM_BASE_URL, LLM_API_KEY)

The spec comes from --llm, else the AGENT_LLM env var, else AGENT_MODEL (an
Anthropic model id, kept for compatibility), else anthropic:claude-opus-5-5.
OpenAI-compatible servers that reject JSON-schema output are retried once in
JSON mode with the schema in the prompt, then validated with Pydantic.
"""
from __future__ import annotations

import json
import os
import re
import warnings
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_SPEC = "anthropic:claude-opus-5-5"
MAX_TOKENS = 4000


class LLMUnavailable(Exception):
    pass


@dataclass(frozen=True)
class Provider:
    kind: str                                   # anthropic | openai | gemini (the SDK used)
    key_env: tuple[str, ...] = ()               # any one of these counts as credentials
    base_url: Optional[str] = None
    base_url_env: Optional[str] = None
    default_model: Optional[str] = None
    needs_key: bool = True
    fixed_key: Optional[str] = None             # servers that ignore the key but need one set


PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider("anthropic", ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE"),
                          default_model="claude-opus-5-5"),
    "openai": Provider("openai", ("OPENAI_API_KEY",), default_model="gpt-6-luna"),
    "gemini": Provider("gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "ollama": Provider("openai", base_url="http://localhost:11434/v1", needs_key=False, fixed_key="ollama"),
    "openrouter": Provider("openai", ("OPENROUTER_API_KEY",), base_url="https://openrouter.ai/api/v1"),
    "deepseek": Provider("openai", ("DEEPSEEK_API_KEY",), base_url="https://api.deepseek.com"),
    "groq": Provider("openai", ("GROQ_API_KEY",), base_url="https://api.groq.com/openai/v1"),
    "compatible": Provider("openai", ("LLM_API_KEY",), base_url_env="LLM_BASE_URL", needs_key=False,
                           fixed_key="none"),
}


def parse_spec(spec: Optional[str] = None) -> tuple[str, str]:
    """'openai:gpt-x' -> ('openai', 'gpt-x'). 'ollama:qwen3:8b' keeps the model's own colon."""
    spec = (spec or os.environ.get("AGENT_LLM") or os.environ.get("AGENT_MODEL") or DEFAULT_SPEC).strip()
    provider, sep, model = spec.partition(":")
    if not sep:
        if spec.startswith("claude-"):          # bare Anthropic model id (old AGENT_MODEL usage)
            return "anthropic", spec
        provider, model = spec, ""
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise LLMUnavailable(f"unknown provider {provider!r}; use one of {', '.join(PROVIDERS)}")
    model = model or PROVIDERS[provider].default_model or ""
    if not model:
        raise LLMUnavailable(f"name a model for {provider}, e.g. --llm {provider}:<model>")
    return provider, model


def has_credentials(provider: str = "anthropic") -> bool:
    p = PROVIDERS[provider]
    if not p.needs_key:
        return p.base_url_env is None or bool(os.environ.get(p.base_url_env))
    return any(os.environ.get(k) for k in p.key_env)


def _api_key(p: Provider) -> Optional[str]:
    return next((os.environ[k] for k in p.key_env if os.environ.get(k)), None) or p.fixed_key


# --------------------------------------------------------------------------- #
class StructuredLLM:
    """Interface: parse(system, user, schema) returns a validated `schema` instance or raises LLMUnavailable."""
    provider = "?"
    model = "?"

    def __init__(self) -> None:
        self.calls = 0
        self.prompt_chars = 0            # input size, to compare context strategies (eval --recent-k / --no-pins)

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def parse(self, system: str, user: str, schema: Type[T]) -> T:  # pragma: no cover - interface
        raise NotImplementedError


class AnthropicLLM(StructuredLLM):
    provider = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", client: Any = None):
        super().__init__()
        try:
            import anthropic
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("anthropic package not installed (pip install anthropic)") from e
        self._sdk = anthropic
        self.client = client or anthropic.Anthropic()
        self.model = model

    def parse(self, system: str, user: str, schema: Type[T]) -> T:
        a = self._sdk
        self.calls += 1
        self.prompt_chars += len(system) + len(user)
        try:
            # `output_format=<Pydantic model>` is what gives us a validated `parsed_output`.
            # Newer SDKs warn that it is deprecated in favour of output_config.format; we
            # silence only that warning until the typed replacement is documented.
            warnings.filterwarnings("ignore", message=".*output_format.*deprecated", category=DeprecationWarning)
            resp = self.client.beta.messages.parse(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                # On a safety decline the API retries on a fallback model in the same call.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.AuthenticationError as e:
            raise LLMUnavailable(f"auth: {e}") from e
        except a.RateLimitError as e:
            raise LLMUnavailable(f"rate limited: {e}") from e
        except a.APIStatusError as e:
            raise LLMUnavailable(f"api status {e.status_code}: {e}") from e
        except a.APIConnectionError as e:
            raise LLMUnavailable(f"connection: {e}") from e
        except ValidationError as e:
            raise LLMUnavailable(f"invalid structured output: {e}") from e

        if resp.stop_reason == "refusal":
            raise LLMUnavailable("model refused")
        if resp.stop_reason == "max_tokens" or resp.parsed_output is None:
            raise LLMUnavailable(f"no parsed output (stop_reason={resp.stop_reason})")
        return resp.parsed_output


def _json_from_text(text: str, schema: Type[T]) -> T:
    """Validate a model's JSON text; tolerates ```json fences some local models add."""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    if not t:
        raise LLMUnavailable("empty output")
    try:
        return schema.model_validate_json(t)
    except ValidationError as e:
        raise LLMUnavailable(f"invalid structured output: {e}") from e


class OpenAILLM(StructuredLLM):
    """OpenAI and every OpenAI-compatible server (Ollama, OpenRouter, DeepSeek, Groq, vLLM, LM Studio...)."""

    def __init__(self, model: str, provider: str = "openai", base_url: Optional[str] = None,
                 api_key: Optional[str] = None, client: Any = None):
        super().__init__()
        try:
            import openai
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("openai package not installed (pip install openai)") from e
        self._sdk = openai
        self.provider = provider
        self.model = model
        try:
            self.client = client or openai.OpenAI(api_key=api_key, base_url=base_url)
        except openai.OpenAIError as e:          # e.g. no OPENAI_API_KEY
            raise LLMUnavailable(f"{provider}: {e}") from e
        self.json_mode = False                   # set once the server rejects JSON-schema output

    def parse(self, system: str, user: str, schema: Type[T]) -> T:
        o = self._sdk
        self.calls += 1
        self.prompt_chars += len(system) + len(user)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            if not self.json_mode:
                try:
                    comp = self.client.chat.completions.parse(model=self.model, messages=messages,
                                                              response_format=schema)
                except o.BadRequestError:
                    self.json_mode = True        # server can't do strict schemas: JSON mode from now on
                else:
                    msg = comp.choices[0].message
                    if getattr(msg, "refusal", None):
                        raise LLMUnavailable("model refused")
                    if msg.parsed is None:
                        raise LLMUnavailable(f"no parsed output (finish_reason={comp.choices[0].finish_reason})")
                    return msg.parsed
            return self._parse_json_mode(messages, schema)
        except o.LengthFinishReasonError as e:
            raise LLMUnavailable("output cut off at the token limit") from e
        except o.ContentFilterFinishReasonError as e:
            raise LLMUnavailable("model refused (content filter)") from e
        except o.AuthenticationError as e:
            raise LLMUnavailable(f"auth: {e}") from e
        except o.RateLimitError as e:
            raise LLMUnavailable(f"rate limited: {e}") from e
        except o.APIStatusError as e:
            raise LLMUnavailable(f"api status {e.status_code}: {e}") from e
        except o.APIConnectionError as e:
            raise LLMUnavailable(f"connection: {e}") from e
        except ValidationError as e:
            raise LLMUnavailable(f"invalid structured output: {e}") from e

    def _parse_json_mode(self, messages: list[dict], schema: Type[T]) -> T:
        system = (messages[0]["content"] + "\n\nReply with ONLY a JSON object that matches this JSON schema:\n"
                  + json.dumps(schema.model_json_schema(), ensure_ascii=False))
        comp = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}, messages[1]],
            response_format={"type": "json_object"})
        choice = comp.choices[0]
        if choice.finish_reason == "length":
            raise LLMUnavailable("output cut off at the token limit")
        return _json_from_text(choice.message.content or "", schema)


class GeminiLLM(StructuredLLM):
    provider = "gemini"

    def __init__(self, model: str, api_key: Optional[str] = None, client: Any = None):
        super().__init__()
        try:
            from google import genai
            from google.genai import errors, types
        except ImportError as e:  # pragma: no cover
            raise LLMUnavailable("google-genai package not installed (pip install google-genai)") from e
        self._types, self._errors = types, errors
        self.model = model
        try:
            self.client = client or genai.Client(api_key=api_key)
        except ValueError as e:                  # no GEMINI_API_KEY / GOOGLE_API_KEY
            raise LLMUnavailable(f"gemini: {e}") from e

    def parse(self, system: str, user: str, schema: Type[T]) -> T:
        import httpx
        self.calls += 1
        self.prompt_chars += len(system) + len(user)
        config = self._types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema.model_json_schema(),
            max_output_tokens=MAX_TOKENS,
        )
        try:
            resp = self.client.models.generate_content(model=self.model, contents=user, config=config)
        except self._errors.ClientError as e:    # 4xx: auth, quota (429), bad request
            raise LLMUnavailable(f"api status {getattr(e, 'code', '?')}: {e}") from e
        except self._errors.APIError as e:       # 5xx and other API errors
            raise LLMUnavailable(f"api error: {e}") from e
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"connection: {e}") from e
        text = resp.text
        if not text:
            cands = getattr(resp, "candidates", None) or []
            reason = getattr(cands[0], "finish_reason", None) if cands else None
            raise LLMUnavailable(f"no output (finish_reason={reason})")
        return _json_from_text(text, schema)


# --------------------------------------------------------------------------- #
def build_llm(spec: Optional[str] = None) -> StructuredLLM:
    provider, model = parse_spec(spec)
    p = PROVIDERS[provider]
    if p.kind == "anthropic":
        return AnthropicLLM(model)
    if p.kind == "gemini":
        return GeminiLLM(model, api_key=_api_key(p))
    base_url = os.environ.get(p.base_url_env) if p.base_url_env else p.base_url
    if p.base_url_env and not base_url:
        raise LLMUnavailable(f"set {p.base_url_env} to the server's OpenAI-compatible URL")
    return OpenAILLM(model, provider=provider, base_url=base_url, api_key=_api_key(p))


def make_llm(mode: str, spec: Optional[str] = None) -> Optional[StructuredLLM]:
    """mode: rules | llm | auto. Returns None when the agent should run rules-only.

    auto uses the LLM only when the chosen provider has credentials (or needs none)."""
    if mode == "rules":
        return None
    try:
        provider, _ = parse_spec(spec)
        if mode == "auto" and not has_credentials(provider):
            return None
        return build_llm(spec)
    except LLMUnavailable:
        if mode == "llm":
            raise
        return None
