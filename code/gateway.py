"""
Research gateway: call a model FAMILY, never a provider.

Contract
--------
1. You ask for a family ("llama"), not a provider.
2. Fallback walks that family's routes only. It never substitutes another
   family. If every route fails, the call raises - a failed observation is
   recorded as failed, not quietly answered by a different model.
3. Every attempt is written to a JSONL trace with the immutable model id that
   was actually sent, the route index, the key index, latency and token
   counts. The trace is the evidence that treatment assignment held.
4. In strict mode (default) a VARIANT route is refused, so a re-quantised
   repack can never stand in for the reference build without being asked for.
5. `judged_by` refuses a judge that shares the producer's publisher, so
   "different-family judging" cannot silently become same-family judging.

This is the P0 fix for the production router, which fell back
deepseek -> mistral -> openrouter:llama and changed the model family mid-run.
"""
from __future__ import annotations

import json
import re
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import registry
from providers import POOL, PROVIDERS
from registry import Family, Fidelity, Route

TRACE_PATH = Path(os.environ.get(
    "SAARTHI_LLM_TRACE",
    Path(__file__).resolve().parent.parent.parent / "Submission" / "llm_trace.jsonl"))


# ---------------------------------------------------------------------------
@dataclass
class LLMResult:
    ok: bool
    text: str
    family: str
    publisher: str
    provider: str
    model_id: str                 # what actually went on the wire
    fidelity: str
    route_index: int
    key_index: int
    attempts: int
    latency_ms: int
    finish_reason: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    role: str = ""
    call_id: str = ""
    error: Optional[str] = None

    @property
    def substituted(self) -> bool:
        """True if a non-primary route served this call (still same family)."""
        return self.route_index > 0


class FamilyExhausted(RuntimeError):
    """Every route for the requested family failed. Do NOT substitute."""


class FidelityViolation(RuntimeError):
    """Only a VARIANT route remains and strict mode forbids it."""


class JudgeIndependenceError(RuntimeError):
    """The requested judge shares a publisher with the producer."""


# ---------------------------------------------------------------------------
class Trace:
    """Append-only JSONL log. One line per ATTEMPT, not per call."""

    def __init__(self, path: Path = TRACE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, rec: dict) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(), **rec}
        with self._lock:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")

    def read(self) -> List[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as fh:
            return [json.loads(l) for l in fh if l.strip()]


TRACE = Trace()

# (provider, model_id, key_index) that refused the MODEL itself (no access,
# no marketplace agreement, unknown id). Retrying cannot help, so the key is
# skipped for the rest of the process and the call moves to the next key or
# the family's fallback route at once. Recorded in the trace.
_NO_ACCESS = ("not available for this account", "invalid_payment_instrument",
              "model access is denied", "use case details have not been submitted",
              "you don't have access to the model", "the provided model identifier is invalid")
DEAD_ROUTE_KEYS: dict = {}          # key -> time marked; expires (AWS agreements propagate)
DEAD_TTL_S = 900.0
#: family -> (provider, model_id): a run can PIN a family to one route (judges
#: are pinned to the route they were qualified on; no silent cross-route mixing)
PINNED_ROUTES: Dict[str, tuple] = {}


def _dead(key) -> bool:
    t = DEAD_ROUTE_KEYS.get(key)
    if t is None:
        return False
    if time.time() - t > DEAD_TTL_S:
        with _DEAD_LOCK:
            DEAD_ROUTE_KEYS.pop(key, None)    # re-probe after the TTL
        return False
    return True
_DEAD_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
class ResearchGateway:
    def __init__(self, strict: bool = True, trace: Trace = TRACE,
                 retries_per_route: int = 2, timeout: float = 90.0):
        self.strict = strict
        self.trace = trace
        self.retries_per_route = retries_per_route
        self.timeout = timeout

    # -- the single entry point -------------------------------------------
    def complete(
        self,
        family: str,
        messages: List[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 900,
        want_json: bool = False,
        role: str = "",
        seed: Optional[int] = None,
        allow_variant: Optional[bool] = None,
        experiment: str = "",
        item_id: str = "",
        timeout: Optional[float] = None,
    ) -> LLMResult:
        fam: Family = registry.get(family)
        allow_variant = (not self.strict) if allow_variant is None else allow_variant
        routes = fam.usable_routes(allow_variant=allow_variant)
        if family in PINNED_ROUTES:
            routes = [r for r in routes if (r.provider, r.model_id) == PINNED_ROUTES[family]]

        if not routes:
            any_route = fam.usable_routes(allow_variant=True)
            if any_route and not allow_variant:
                raise FidelityViolation(
                    f"family {family!r}: only VARIANT routes are reachable "
                    f"({', '.join(str(r) for r in any_route)}). Strict mode refuses a "
                    f"different build as a stand-in. Pass allow_variant=True to accept "
                    f"it, and report it as a different model.")
            raise FamilyExhausted(f"family {family!r}: no reachable route")

        call_id = uuid.uuid4().hex[:12]
        last_err: Optional[Exception] = None
        attempts = 0

        for ridx, route in enumerate(routes):
            n_keys = max(1, len(PROVIDERS[route.provider].keys))
            # rotate keys WITHIN this call so every key is tried before falling
            # back to the next route (a shared counter let other threads steer
            # all of one call's retries to the same account)
            start = POOL.next_key_index(route.provider)
            for k in range(n_keys * self.retries_per_route):
                kidx = (start + k) % n_keys
                if _dead((route.provider, route.model_id, kidx)):
                    if all(_dead((route.provider, route.model_id, i)) for i in range(n_keys)):
                        break             # no key can serve this model: next route
                    continue
                attempts += 1
                t0 = time.time()
                try:
                    res = self._one(route, kidx, messages, temperature, max_tokens,
                                    want_json, seed, timeout or self.timeout)
                    out = LLMResult(
                        ok=True, text=res["text"], family=fam.key,
                        publisher=fam.publisher, provider=route.provider,
                        model_id=route.model_id, fidelity=route.fidelity.value,
                        route_index=ridx, key_index=kidx, attempts=attempts,
                        latency_ms=int((time.time() - t0) * 1000),
                        finish_reason=res["finish"], prompt_tokens=res["pt"],
                        completion_tokens=res["ct"], role=role, call_id=call_id)
                    self._log(out, experiment, item_id, seed, temperature)
                    return out
                except Exception as e:  # noqa: BLE001 - every failure is data
                    last_err = e
                    if any(t in str(e).lower() for t in _NO_ACCESS):
                        with _DEAD_LOCK:
                            DEAD_ROUTE_KEYS[(route.provider, route.model_id, kidx)] = time.time()
                    self.trace.write({
                        "call_id": call_id, "experiment": experiment,
                        "item_id": item_id, "role": role, "ok": False,
                        "family": fam.key, "publisher": fam.publisher,
                        "provider": route.provider, "model_id": route.model_id,
                        "fidelity": route.fidelity.value, "route_index": ridx,
                        "key_index": kidx, "attempt": attempts,
                        "latency_ms": int((time.time() - t0) * 1000),
                        "error": str(e)[:300],
                        "no_access": any(t in str(e).lower() for t in _NO_ACCESS),
                    })
                    if any(t in str(e).lower() for t in _NO_ACCESS):
                        continue          # no backoff: this key cannot serve the model
                    # Throttling is the provider asking us to slow down, not a
                    # dead route: wait longer before rotating key / route, so a
                    # family is not pushed onto its fallback by our own burst.
                    m = str(e).lower()
                    throttled = ("429" in m or "throttl" in m or "too many requests" in m
                                 or "rate limit" in m)
                    time.sleep(min(5 * 2 ** (attempts - 1), 40) if throttled
                               else min(2 ** (attempts - 1), 8))

        raise FamilyExhausted(
            f"family {family!r}: all {len(routes)} route(s) failed after "
            f"{attempts} attempts. Last error: {last_err}. No substitution made.")

    # -- judge with an enforced independence check -------------------------
    def judged_by(self, judge_family: str, producer_family: str,
                  messages: List[dict], **kw) -> LLMResult:
        pj, pp = registry.get(judge_family), registry.get(producer_family)
        if pj.publisher == pp.publisher:
            raise JudgeIndependenceError(
                f"judge {judge_family!r} and producer {producer_family!r} share "
                f"publisher {pj.publisher!r}; this is not different-family judging. "
                f"Use a judge from another publisher, or record the condition as "
                f"same-publisher explicitly.")
        kw.setdefault("role", f"judge[{judge_family}<-{producer_family}]")
        return self.complete(judge_family, messages, **kw)

    # -- internals ---------------------------------------------------------
    def _one(self, route: Route, key_index: int, messages, temperature,
             max_tokens, want_json, seed, timeout) -> dict:
        client = POOL.client(route.provider, key_index, timeout=timeout)
        kwargs = dict(model=route.model_id, messages=messages,
                      temperature=temperature, max_tokens=max_tokens)
        mid = route.model_id.lower()
        # GPT-5 family: only the default temperature is accepted, and the
        # output budget is max_completion_tokens (it also covers reasoning).
        if "claude-sonnet-5" in mid:
            # Anthropic rejects `temperature` for Sonnet 5 ('deprecated for this
            # model'); omitted on every route so primary and fallback decode alike
            kwargs.pop("temperature", None)
        if re.search(r"(^|[./])gpt-5", mid):          # also Bedrock 'us.openai.gpt-5.6-luna'
            kwargs.pop("temperature", None)
            kwargs["max_completion_tokens"] = max(kwargs.pop("max_tokens"), 2000)
        # Reasoning models spend tokens before answering; with a small budget
        # they return content=None, finish_reason="length". Give them room.
        if (mid.startswith("sarvam") or "kimi" in mid or "deepseek.v3" in mid or mid.startswith("grok-4.5")
                or mid.startswith("grok-4.6")
                or ("gemini-3" in mid and "pro" in mid) or "claude-sonnet-5" in mid):
            kwargs["max_tokens"] = max(kwargs["max_tokens"], 4000)
        if seed is not None:
            kwargs["seed"] = seed
        if want_json:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            r = client.chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            # some endpoints reject response_format or seed; retry once without
            drop = [k for k, tok in (("response_format", "response_format"),
                                     ("response_format", "json_object"),
                                     ("seed", "seed")) if tok in msg and k in kwargs]
            if drop:
                for k in set(drop):
                    kwargs.pop(k, None)
                r = client.chat.completions.create(**kwargs)
            else:
                raise
        ch = r.choices[0]
        text = (ch.message.content or "").strip()
        finish = getattr(ch, "finish_reason", None)      # kept as given; None stays None
        from json_parse import classify_finish
        if not text and classify_finish(finish) not in ("truncated", "filtered"):
            raise RuntimeError("empty content returned")
        # an empty reply cut off at the token limit is an observation about the
        # model (reasoning used the budget), not an outage: returned, not retried
        u = getattr(r, "usage", None)
        return {"text": text,
                "finish": finish,
                "pt": getattr(u, "prompt_tokens", 0) or 0,
                "ct": getattr(u, "completion_tokens", 0) or 0}

    def _log(self, r: LLMResult, experiment: str, item_id: str,
             seed: Optional[int], temperature: float) -> None:
        d = asdict(r)
        d.pop("text", None)                       # keep the trace small
        d.update({"experiment": experiment, "item_id": item_id,
                  "seed": seed, "temperature": temperature,
                  "text_chars": len(r.text), "substituted": r.substituted})
        self.trace.write(d)


GATEWAY = ResearchGateway()


# ---------------------------------------------------------------------------
def trace_report(path: Path = TRACE_PATH) -> dict:
    """Summarise a trace: did any treatment get served off its primary route?"""
    rows = Trace(path).read()
    ok = [r for r in rows if r.get("ok")]
    subs = [r for r in ok if r.get("route_index", 0) > 0]
    per_family: Dict[str, dict] = {}
    for r in ok:
        f = r.get("family", "?")
        d = per_family.setdefault(f, {"calls": 0, "off_primary": 0, "models": set(),
                                      "prompt_tokens": 0, "completion_tokens": 0})
        d["calls"] += 1
        d["off_primary"] += 1 if r.get("route_index", 0) > 0 else 0
        d["models"].add(r.get("model_id"))
        d["prompt_tokens"] += r.get("prompt_tokens", 0) or 0
        d["completion_tokens"] += r.get("completion_tokens", 0) or 0
    for d in per_family.values():
        d["models"] = sorted(x for x in d["models"] if x)
    return {"attempts": len(rows), "successful": len(ok),
            "failed": len(rows) - len(ok), "off_primary_route": len(subs),
            "per_family": per_family}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        print(json.dumps(trace_report(), indent=2, default=str))
    else:
        r = GATEWAY.complete(
            "llama",
            [{"role": "user", "content": "Reply with exactly: ok"}],
            max_tokens=8, role="selftest", experiment="smoke")
        print(f"family={r.family} via {r.provider}:{r.model_id} "
              f"[{r.fidelity}] route#{r.route_index} {r.latency_ms}ms -> {r.text!r}")
