"""
Offline stand-in for the gateway: dry runs and unit tests make NO API calls.

    with fakes.patched(fakes.FakeGateway(responder)):
        ...   # every GATEWAY.complete / judged_by in the study code hits the responder

`responder(family, messages) -> str` returns the model's text. The fake keeps
the gateway's contract (LLMResult fields, publisher-independence check), so a
dry run exercises the same parsing, validation and bookkeeping as a live run.
"""
from __future__ import annotations

import contextlib
import itertools
from typing import Callable, List

import gateway as G
import registry

_ids = itertools.count(1)


class FakeGateway:
    def __init__(self, responder: Callable[[str, List[dict]], str], finish_reason: str = "stop"):
        self.responder = responder
        self.finish_reason = finish_reason
        self.calls: List[dict] = []

    def complete(self, family, messages, *, max_tokens=900, role="", **kw) -> G.LLMResult:
        fam = registry.get(family)
        text = self.responder(family, messages)
        p_tok = sum(len(str(m.get("content", ""))) for m in messages) // 4
        self.calls.append({"family": family, "role": role, "prompt_tokens": p_tok,
                           "completion_tokens": len(text) // 4, "max_tokens": max_tokens})
        return G.LLMResult(ok=True, text=text, family=family, publisher=fam.publisher, provider="fake",
                           model_id=f"fake:{family}", fidelity="exact", route_index=0, key_index=0,
                           attempts=1, latency_ms=1, finish_reason=self.finish_reason,
                           prompt_tokens=p_tok, completion_tokens=len(text) // 4, role=role,
                           call_id=f"fake-{next(_ids)}")

    def judged_by(self, judge_family, producer_family, messages, **kw) -> G.LLMResult:
        pj, pp = registry.get(judge_family), registry.get(producer_family)
        if pj.publisher == pp.publisher:
            raise G.JudgeIndependenceError(f"{judge_family} and {producer_family} share {pj.publisher}")
        return self.complete(judge_family, messages, **kw)


@contextlib.contextmanager
def patched(fake: FakeGateway):
    """Route every study module's GATEWAY to the fake for the duration."""
    import verify
    saved = (G.GATEWAY, verify.GATEWAY)
    G.GATEWAY = fake
    verify.GATEWAY = fake
    try:
        yield fake
    finally:
        G.GATEWAY, verify.GATEWAY = saved
