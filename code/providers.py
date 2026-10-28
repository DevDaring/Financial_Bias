"""
Provider layer: transports only.

A *provider* is a way to reach a model over the wire (an OpenAI-compatible
endpoint plus one or more API keys). It is deliberately NOT a research
identity. The research identity is the model family, which lives in
registry.py.

This split exists because the production gateway conflated the two: its
fallback chain moved from `deepseek` to `mistral` to `openrouter:llama`, so a
rate-limit on one provider silently changed the model family mid-experiment.
For a controlled experiment that is a confound, not a resilience feature.

Keys are read from the repository .env. Nothing here selects a model.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# --------------------------------------------------------------------------
# .env discovery (same search order the production app uses)
# --------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_CANDIDATES = [
    _HERE.parent / ".env",                      # Codes/.env
    _HERE.parent.parent / "Codes" / ".env",
    _HERE.parent.parent / ".env",
    _HERE.parent.parent / "saarthi" / "backend" / ".env",
]
ENV_PATH: Optional[Path] = next((p for p in _CANDIDATES if p.exists()), None)
if ENV_PATH is not None:
    load_dotenv(ENV_PATH)


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v is not None and str(v).strip():
            return str(v).strip().strip('"').strip("'")
    return default


def _keys(*names: str) -> List[str]:
    out: List[str] = []
    for n in names:
        v = os.getenv(n)
        if v and str(v).strip():
            k = str(v).strip().strip('"').strip("'")
            if k not in out:
                out.append(k)
    return out


# --------------------------------------------------------------------------
@dataclass
class Provider:
    """One OpenAI-compatible endpoint with a pool of keys."""
    name: str
    base_url: str
    keys: List[str] = field(default_factory=list)
    extra_headers: Dict[str, str] = field(default_factory=dict)
    # providers that aggregate many publishers' models (routers)
    is_router: bool = False

    @property
    def available(self) -> bool:
        return bool(self.keys) and bool(self.base_url)

    def __repr__(self) -> str:
        return f"<Provider {self.name} keys={len(self.keys)} router={self.is_router}>"


def build_providers() -> Dict[str, Provider]:
    """Read every provider this project can reach. No model choices here."""
    p: Dict[str, Provider] = {}

    p["deepseek"] = Provider(
        name="deepseek",
        base_url=_env("DEEPSEEK_API_BASE_URL", "DEEPSEEK_BASE_URL",
                      default="https://api.deepseek.com/v1"),
        keys=_keys("DEEPSEEK_API_KEY_1", "DEEPSEEK_API_KEY_2", "DEEPSEEK_API_KEY"),
    )

    p["mistral"] = Provider(
        name="mistral",
        base_url=_env("MISTRAL_API_BASE_URL", "MISTRAL_BASE_URL",
                      default="https://api.mistral.ai/v1"),
        keys=_keys("MISTRAL_API_KEY1", "MISTRAL_API_KEY2",
                   "MISTRAL_API_KEY_1", "MISTRAL_API_KEY_2", "MISTRAL_API_KEY"),
    )

    p["gemini"] = Provider(
        name="gemini",
        base_url=_env("GEMINI_API_BASE_URL",
                      default="https://generativelanguage.googleapis.com/v1beta/openai"),
        # keys 1 is an OAuth token in this .env; keep only real AIza* keys
        keys=[k for k in _keys("GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
                               "GEMINI_API_KEY_4", "GEMINI_API_KEY_1")
              if k.startswith("AIza")],
    )

    # NanoGPT: several keys exist in .env; NanoGPT_88_API_Key is the funded one
    # and must be tried first. Order here IS the rotation order.
    p["nanogpt"] = Provider(
        name="nanogpt",
        # NOTE: .env also defines NanoGPT_Base_URL=https://api.nanogpt.com/v1,
        # which does not resolve. Nano_GPT_Base_URL is the working endpoint.
        base_url=_env("Nano_GPT_Base_URL", "NANOGPT_BASE_URL",
                      default="https://nano-gpt.com/api/v1"),
        keys=_keys("NanoGPT_88_API_Key", "NanoGPT_88_API_KEY",
                   "NanoGPT_2009_API_KEY", "Nano_GPT_API_KEY", "NANOGPT_API_KEY"),
        is_router=True,
    )

    p["openrouter"] = Provider(
        name="openrouter",
        base_url=_env("OPENROUTER_API_BASE_URL", "OPENROUTER_BASE_URL",
                      default="https://openrouter.ai/api/v1"),
        keys=_keys("OPENROUTER_API_KEY_1", "OPENROUTER_API_KEY_2", "OPENROUTER_API_KEY"),
        extra_headers={"HTTP-Referer": "https://koushikdeb.duckdns.org",
                       "X-Title": "SAARTHI-Research"},
        is_router=True,
    )

    # --- AWS Bedrock: two accounts, rotated like keys. Each "key" here is
    # "ACCESS:SECRET"; the client factory splits it. Region us-east-1 serves
    # every model in the study, including the us.* Llama 4 inference profile.
    pairs = []
    for ak, sk in (("AWS_ACCESS_KEY", "AWS_SECRET_KEY"),
                   ("AWS_ACCESS_KEY2", "AWS_SECRET_KEY2")):
        a, s_ = _env(ak), _env(sk)
        if a and s_ and f"{a}:{s_}" not in pairs:
            pairs.append(f"{a}:{s_}")
    p["bedrock"] = Provider(
        name="bedrock",
        base_url=_env("AWS_BEDROCK_REGION", default="us-east-1"),
        keys=pairs,
    )

    # --- Sarvam AI (India). OpenAI-compatible chat endpoint. sarvam-105b is a
    # reasoning model: with a small output budget it returns content=None and
    # finish_reason="length", so the gateway raises its budget (see gateway).
    p["sarvam"] = Provider(
        name="sarvam",
        base_url=_env("SARVAM_BASE_URL", default="https://api.sarvam.ai/v1"),
        keys=_keys("SARVAM_88_API_KEY", "SARVAM_2009_API_KEY", "SARVAM_PHD_API_KEY"),
    )

    # --- LinkAPI (OpenAI-compatible aggregator) for GPT-5-mini.
    # Verified endpoint: https://api.linkapi.ai/v1 (api.linkapi.org rejects the key).
    p["linkapi"] = Provider(
        name="linkapi",
        base_url=_env("LINKAPI_BASE_URL", default="https://api.linkapi.ai/v1"),
        keys=_keys("LINKAPI_AzureOpenAI_KEY"),
        is_router=True,
    )

    # --- LinkAPI Grok channel. Keys are scoped per model group: this key gets
    # 503 on gpt-5-mini and the Azure key gets 503 on grok, so it is a
    # separate provider. On 2 Oct 2026 grok-4.5/4.6/4.7 accepted the request
    # but never answered (no headers in 240 s) - channel stalled upstream.
    # GrokHeavy_2009_API_KEY answers all three in 2-4 s (same day), so it is
    # the only key. GROKBUILD still hangs 90 s, so it is not rotated in.
    p["linkapi_grok"] = Provider(
        name="linkapi_grok",
        base_url=_env("LINKAPI_BASE_URL", default="https://api.linkapi.ai/v1"),
        keys=_keys("LINKAPI_GROKHEAVY_KEY"),     # 7 Oct 2026: author's instruction - Grok via LinkAPI only
        is_router=True,
    )
    # --- LinkAPI GrokBuild group (same Grok models, group ratio 0.05 vs 0.25 for GrokHeavy;
    # 7 Oct probe: grok-4.5 answers in ~30 s with >1,300 reasoning tokens): same-model fallback only.
    p["linkapi_grokbuild"] = Provider(
        name="linkapi_grokbuild",
        base_url=_env("LINKAPI_BASE_URL", default="https://api.linkapi.ai/v1"),
        keys=_keys("LINKAPI_GROKBUILD_KEY"),
        is_router=True,
    )

    # --- self-hosted vLLM (Vast.ai GPU, 7 Oct 2026): open-weight judge with token log-probabilities.
    # URL and key are set per session in the environment (VAST_VLLM_URL, VAST_VLLM_KEY).
    if _env("VAST_VLLM_URL"):
        p["vast_vllm"] = Provider(name="vast_vllm", base_url=_env("VAST_VLLM_URL"), keys=_keys("VAST_VLLM_KEY"),
                                  is_router=False)

    # --- LinkAPI Gemini channel (separate model group, separate key).
    p["linkapi_gemini"] = Provider(
        name="linkapi_gemini",
        base_url=_env("LINKAPI_BASE_URL", default="https://api.linkapi.ai/v1"),
        keys=_keys("LINKAPI_Gemini_Cheap_KEY"),
        is_router=True,
    )

    # --- direct judge platforms (2 Oct 2026): the author's own keys; OpenRouter
    # is the fallback route in the registry
    if _keys("Calude_2009_API_Key"):
        p["anthropic"] = Provider(
            name="anthropic", base_url="https://api.anthropic.com/v1",
            keys=_keys("Calude_2009_API_Key"),
            # the key is not scoped to a workspace: every request names one
            extra_headers={"anthropic-workspace-id": _env("Claude_workspace_id", default="")})
    if _keys("Open_AI_2009_Key"):
        p["openai_direct"] = Provider(
            name="openai_direct", base_url="https://api.openai.com/v1", keys=_keys("Open_AI_2009_Key"))
    if _keys("Grok_API_KEY"):
        p["xai_direct"] = Provider(
            name="xai_direct", base_url="https://api.x.ai/v1", keys=_keys("Grok_API_KEY"))

    # optional direct endpoints; used only if the registry references them
    if _keys("XAI_API_KEY", "GROK_API_KEY"):
        p["xai"] = Provider(
            name="xai",
            base_url=_env("XAI_API_BASE_URL", default="https://api.x.ai/v1"),
            keys=_keys("XAI_API_KEY", "GROK_API_KEY"),
        )
    if _keys("OPENAI_API_KEY", "OPENAI_API_KEY_1"):
        p["openai"] = Provider(
            name="openai",
            base_url=_env("OPENAI_API_BASE_URL", default="https://api.openai.com/v1"),
            keys=_keys("OPENAI_API_KEY", "OPENAI_API_KEY_1"),
        )

    return p


PROVIDERS: Dict[str, Provider] = build_providers()


# --------------------------------------------------------------------------
# client cache + round-robin key rotation (per provider)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Bedrock shim: makes boto3's Converse API look like openai's
# client.chat.completions.create(...) so the gateway needs no special case.
# --------------------------------------------------------------------------
class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class AnthropicChat:
    """Anthropic Messages API behind the OpenAI chat.completions interface the
    gateway uses (no SDK dependency). System messages go to `system`;
    stop_reason is mapped to the OpenAI finish_reason vocabulary."""

    def __init__(self, key: str, base_url: str, headers: Optional[dict], timeout: float = 90.0):
        self.key, self.base, self.headers, self.timeout = key, base_url.rstrip("/"), headers or {}, timeout
        self.chat = _NS(completions=_NS(create=self._create))

    def with_options(self, **_):
        return self

    def _create(self, model: str, messages: list, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None, max_completion_tokens: Optional[int] = None, **_ignored):
        import json as _json
        import urllib.error
        import urllib.request
        body = {"model": model, "max_tokens": int(max_completion_tokens or max_tokens or 900),
                "messages": [{"role": m["role"], "content": m["content"]} for m in messages
                             if m["role"] in ("user", "assistant")]}
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if system:
            body["system"] = system
        if temperature is not None:
            body["temperature"] = temperature
        req = urllib.request.Request(
            f"{self.base}/messages", data=_json.dumps(body).encode(),
            headers={"x-api-key": self.key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json", **{k: v for k, v in self.headers.items() if v}})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as f:
                r = _json.load(f)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Error code: {e.code} - {e.read()[:300].decode(errors='ignore')}") from None
        text = "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
        stop = r.get("stop_reason")
        finish = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                  "refusal": "content_filter", "tool_use": "tool_calls"}.get(stop, stop)
        u = r.get("usage", {})
        return _NS(choices=[_NS(message=_NS(content=text), finish_reason=finish)],
                   usage=_NS(prompt_tokens=u.get("input_tokens", 0),
                             completion_tokens=u.get("output_tokens", 0)),
                   model=r.get("model"))


class BedrockChat:
    def __init__(self, access_key: str, secret_key: str, region: str = "us-east-1",
                 timeout: float = 90.0):
        import boto3
        from botocore.config import Config
        self._c = boto3.client(
            "bedrock-runtime", region_name=region,
            aws_access_key_id=access_key, aws_secret_access_key=secret_key,
            config=Config(read_timeout=timeout, connect_timeout=20,
                          retries={"max_attempts": 0}))
        self.chat = _NS(completions=_NS(create=self._create))

    def with_options(self, **_):
        return self

    def _create(self, model: str, messages: list, temperature: Optional[float] = None,
                max_tokens: Optional[int] = None, max_completion_tokens: Optional[int] = None,
                **_ignored):
        # GPT-5 style calls send max_completion_tokens and NO temperature (only the
        # default is accepted): honour both, so the Bedrock route decodes like the
        # OpenAI-compatible route the judge was qualified on
        max_tokens = max_completion_tokens or max_tokens or 900
        # response_format / seed have no Converse equivalent; JSON is requested
        # in the prompt text instead, and the trace records the seed it asked for.
        system = [{"text": m["content"]} for m in messages if m["role"] == "system"]
        conv = [{"role": m["role"], "content": [{"text": m["content"]}]}
                for m in messages if m["role"] in ("user", "assistant")]
        kw = dict(modelId=model, messages=conv,
                  inferenceConfig=({"maxTokens": max_tokens, "temperature": temperature}
                                   if temperature is not None else {"maxTokens": max_tokens}))
        if system:
            kw["system"] = system
        r = self._c.converse(**kw)
        parts = r["output"]["message"]["content"]
        text = "".join(p.get("text", "") for p in parts if "text" in p)
        u = r.get("usage", {})
        stop = r.get("stopReason")             # missing stays missing (not 'clean')
        return _NS(choices=[_NS(message=_NS(content=text),
                                finish_reason={"max_tokens": "length",
                                               "model_context_window_exceeded": "length",
                                               "content_filtered": "content_filter",
                                               "guardrail_intervened": "content_filter",
                                               "end_turn": "stop", "stop_sequence": "stop",
                                               "tool_use": "tool_calls",
                                               "malformed_model_output": "error",
                                               "malformed_tool_use": "error"}.get(stop, stop))],
                   usage=_NS(prompt_tokens=u.get("inputTokens", 0),
                             completion_tokens=u.get("outputTokens", 0)))


class _ClientPool:
    def __init__(self):
        self._clients: Dict[tuple, object] = {}
        self._rr: Dict[str, int] = {}
        self._lock = threading.Lock()

    def next_key_index(self, provider: str) -> int:
        with self._lock:
            n = len(PROVIDERS[provider].keys)
            if n == 0:
                return 0
            i = self._rr.get(provider, 0) % n
            self._rr[provider] = (i + 1) % n
            return i

    def client(self, provider: str, key_index: int, timeout: float = 90.0):
        from openai import OpenAI
        k = (provider, key_index, timeout)
        with self._lock:
            c = self._clients.get(k)
            if c is None and provider == "anthropic":
                pr = PROVIDERS[provider]
                c = AnthropicChat(pr.keys[key_index], pr.base_url, pr.extra_headers, timeout=timeout)
                self._clients[k] = c
            if c is None and provider == "bedrock":
                pr = PROVIDERS[provider]
                ak, sk = pr.keys[key_index].split(":", 1)
                c = BedrockChat(ak, sk, region=pr.base_url, timeout=timeout)
                self._clients[k] = c
            if c is None:
                pr = PROVIDERS[provider]
                c = OpenAI(
                    base_url=pr.base_url,
                    api_key=pr.keys[key_index],
                    timeout=timeout,
                    max_retries=0,   # retries are managed by the gateway
                    default_headers=pr.extra_headers or None,
                )
                self._clients[k] = c
            return c


POOL = _ClientPool()


# --------------------------------------------------------------------------
# key health check
# --------------------------------------------------------------------------
_PROBE_MODEL = {
    "deepseek": "deepseek-chat",
    "mistral": "mistral-small-latest",
    "gemini": "gemini-2.5-flash-lite",
    "nanogpt": "meta-llama/llama-3.1-8b-instruct",
    "openrouter": "meta-llama/llama-3.1-8b-instruct",
    "xai": "grok-4.3",
    "openai": "gpt-4o-mini",
    "bedrock": "amazon.nova-lite-v1:0",
    "sarvam": "sarvam-105b",
    "linkapi": "gpt-5-mini",
    "anthropic": "claude-sonnet-5",
    "openai_direct": "gpt-5.6-luna",
    "xai_direct": "grok-4.20-0309-non-reasoning",
    "linkapi_gemini": "gemini-2.5-flash",
    "linkapi_grok": "grok-4.5",
    "linkapi_grokbuild": "grok-4.5",
}


def prune_dead_keys(providers: Optional[List[str]] = None,
                    timeout: float = 25.0, verbose: bool = True) -> Dict[str, dict]:
    """Probe every key once and drop the dead ones from rotation.

    Without this, round-robin sprays calls at revoked keys: in the first live
    pilot 227 of 565 attempts (40%) failed purely because 2 of 3 NanoGPT keys
    and 2 of 4 Gemini keys were revoked. The retry logic recovered every call,
    so no data was lost - but it doubled latency and burned the budget.
    """
    names = providers or list(PROVIDERS)
    out: Dict[str, dict] = {}
    for name in names:
        p = PROVIDERS.get(name)
        if p is None or not p.keys:
            continue
        live, dead = [], []
        for i, k in enumerate(list(p.keys)):
            try:
                c = POOL.client(name, i, timeout=timeout)
                c.chat.completions.create(
                    model=_PROBE_MODEL.get(name, "gpt-4o-mini"),
                    messages=[{"role": "user", "content": "hi"}], max_tokens=2)
                live.append(k)
            except Exception as e:
                msg = str(e)
                # only drop on auth/permission failures; a rate limit or a
                # transient 5xx is not a dead key
                if any(s in msg for s in ("401", "403", "invalid_api_key",
                                          "Invalid session", "valid API key",
                                          "Unauthorized")):
                    dead.append((i, msg[:70]))
                else:
                    live.append(k)
        if dead:
            p.keys = live
            POOL._clients = {k: v for k, v in POOL._clients.items() if k[0] != name}
            POOL._rr[name] = 0
        out[name] = {"live": len(live), "dead": len(dead),
                     "dead_detail": [d[1] for d in dead]}
        if verbose:
            flag = "OK " if live else "OFF"
            extra = f"  dropped {len(dead)} dead" if dead else ""
            print(f"  [{flag}] {name:<11} {len(live)} live key(s){extra}")
    return out


def summary() -> List[dict]:
    """Non-secret provider summary (key counts only, never key values)."""
    return [{"name": p.name, "base_url": p.base_url, "n_keys": len(p.keys),
             "router": p.is_router, "available": p.available}
            for p in PROVIDERS.values()]


if __name__ == "__main__":
    print(f".env: {ENV_PATH}")
    for s in summary():
        flag = "ok " if s["available"] else "OFF"
        print(f"  [{flag}] {s['name']:<11} keys={s['n_keys']}  router={str(s['router']):<5} {s['base_url']}")
