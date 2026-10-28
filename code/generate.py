"""
Generation arms (plan §4.2).

Three comparators, all reading the SAME evidence object:

  template    deterministic, no LLM. The fallback AND a substantive baseline -
              the plan is explicit that a safe template must be a real
              comparator, not a straw man.
  grounded    LLM sees the evidence and is told to phrase only what is there.
              This is roughly what the production app does today.
  typed       LLM sees the typed contract, must emit claims that each cite
              evidence ids, and is told the permitted statement per type.
              This is the proposed method.

E2 cue conditions are handled here too: a cue is injected explicitly and
labelled as an experimental cue, never as the borrower's factual attribute,
and never fed to the predictor.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from evidence import (FORBIDDEN_CAUSAL_TERMS, FORBIDDEN_TREND_TERMS,
                      REASON_CONCEPTS, EvidenceObject, EvidenceType)
from feature_semantics import REASON_CONCEPTS_V2
from gateway import GATEWAY, LLMResult

ARMS = ("template", "grounded", "typed")


@dataclass
class Candidate:
    """One generated explanation, before verification."""
    borrower_id: str
    arm: str
    text: str
    claims: List[dict] = field(default_factory=list)   # typed arm only
    family: str = "template"
    model_id: str = "deterministic"
    provider: str = "-"
    route_index: int = 0
    latency_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cue: Optional[str] = None
    repetition: int = 0
    error: Optional[str] = None
    raw: str = ""
    parse_method: str = ""          # direct | repair | parser | truncated | ... (json_parse)
    parse_info: dict = field(default_factory=dict)
    call_id: str = ""
    finish_reason: Optional[str] = None
    seed: Optional[int] = None
    temperature: Optional[float] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


# ---------------------------------------------------------------------------
# 1. deterministic template  (fallback + baseline)
# ---------------------------------------------------------------------------
def render_template(eo: EvidenceObject, max_drivers: int = 4) -> Candidate:
    """Readable prose from evidence alone. No model, no invention."""
    facts = eo.by_type(EvidenceType.OBSERVED_FACT)
    attrs = sorted(eo.by_type(EvidenceType.MODEL_ATTRIBUTION),
                   key=lambda a: -abs(a.contribution or 0.0))[:max_drivers]
    pred = eo.prediction
    unknown = eo.by_type(EvidenceType.UNSUPPORTED)

    parts: List[str] = []
    if pred is not None:
        parts.append(
            f"For the documented outcome ({pred.target_definition}, horizon: "
            f"{pred.horizon}), the model estimates a probability of "
            f"{pred.calibrated_pd:.0%}.")
    if facts:
        bits = []
        for f in facts[:max_drivers]:
            v = f"{f.value:,.2f}" if isinstance(f.value, float) else f"{f.value:,}" \
                if isinstance(f.value, int) else str(f.value)
            bits.append(f"{f.source_field} = {v}{(' ' + f.unit) if f.unit else ''}")
        parts.append("Recorded values at the observation time: " + "; ".join(bits) + ".")
    if attrs:
        up = [a for a in attrs if a.direction == "increases_risk"]
        dn = [a for a in attrs if a.direction == "decreases_risk"]
        seg = []
        if up:
            seg.append("raise the model's risk score: "
                       + ", ".join(a.feature_group for a in up))
        if dn:
            seg.append("lower it: " + ", ".join(a.feature_group for a in dn))
        parts.append(
            "Relative to the model's reference background, these features "
            + "; ".join(seg) + ". These are contributions on the "
            f"{attrs[0].explained_scale}, not changes in the probability itself.")
    if unknown:
        parts.append("No supported estimate is available for: "
                     + "; ".join(u.reason or "unspecified item" for u in unknown) + ".")
    parts.append("This describes the model's behaviour, not a verified cause of default.")
    return Candidate(borrower_id=eo.borrower_id, arm="template",
                     text=" ".join(parts),
                     claims=[{"text": p, "evidence_ids": eo.ids()} for p in parts])


# ---------------------------------------------------------------------------
# 2. prompts
# ---------------------------------------------------------------------------
_RULES = f"""Rules you must follow:
1. Use ONLY the supplied evidence. Never introduce a factor that is not there.
2. Never contradict a stated direction.
3. Never state a probability of your own; only the supplied calibrated estimate.
4. Describe CURRENT LEVELS. You are seeing one snapshot, so never assert a
   change over time. Forbidden words include: {', '.join(FORBIDDEN_TREND_TERMS[:12])}.
5. Never assert real-world causation from a modelled scenario. Forbidden:
   {', '.join(FORBIDDEN_CAUSAL_TERMS[:8])}.
6. Attributions are contributions on the model's own scale, not changes in the
   probability. Do not convert them into percentage-point effects.
7. If an item is marked unsupported, say that no supported estimate is
   available. Do not fill the gap."""

# ---- producer/2.0 (2 Oct 2026): aligned with the repaired rubric (plan §5.2)
# and evidence/2.0. v1 prompts above are kept verbatim for the saved pilot.
_RULES_V2 = f"""Rules you must follow:
1. Use ONLY the supplied evidence. Never introduce a factor, fact or reason that is not there.
2. Never contradict a stated direction. Attributions say whether a feature RAISES or LOWERS
   the model's risk score relative to its reference.
3. Never state a probability of your own; only the supplied calibrated estimate.
4. A number you write must be the value of the field you attach it to (sensible rounding is
   fine). Attribution contributions are on the model's raw log-odds scale: never present
   them as percentages or as changes in the probability.
5. Describe recorded values. You may compare two periods of the same field only if both
   values are in the evidence and you name both periods. Never assert any other change
   over time. Forbidden otherwise: {', '.join(FORBIDDEN_TREND_TERMS[:10])}.
6. Never assert a real-world cause, and never say an action will change the outcome.
   Forbidden: {', '.join(FORBIDDEN_CAUSAL_TERMS[:8])}.
7. For an encoded field, state the recorded value with its documented label. If the label
   says the code is undocumented, say so. A field name alone is not the borrower's category.
8. If a value is not recorded, say so. Never treat a missing value as zero.
9. Mention the estimated probability and the strongest driver."""

_GROUNDED_SYS_V2 = (
    "You are a credit analyst writing a short explanation of a model's output "
    "for a credit officer.\n" + _RULES_V2 +
    "\nWrite 3-5 plain sentences. "
    'Output strict JSON only: {"explanation":"<the 3-5 sentences>"}')

#: fresh-test prompt style (9 Oct 2026): written to the APPLICANT in everyday language, same rules
_APPLICANT_SYS_V2 = (
    "You are writing to the loan applicant to explain, in everyday language, what the bank's credit model "
    "based its assessment on. Avoid jargon and internal field codes; describe each factor in plain words.\n"
    + _RULES_V2 +
    "\nWrite 3-5 plain sentences. "
    'Output strict JSON only: {"explanation":"<the 3-5 sentences>"}')

_TYPED_SYS_V2 = (
    "You are a credit analyst writing a short explanation of a model's output "
    "for a credit officer.\n" + _RULES_V2 +
    "\nEvery claim you make must cite the evidence ids it rests on.\n"
    'Output strict JSON: {"claims":[{"text":"<one sentence>",'
    '"evidence_ids":["f1","a1"],"concept":"<one allowed concept or null>"}]}\n'
    f"Allowed concepts: {list(REASON_CONCEPTS_V2)}")

PROMPT_VERSION = "producer/2.1"   # 2.1: grounded arm also answers in JSON

_GROUNDED_SYS = (
    "You are a credit analyst writing a short explanation of a model's output "
    "for a credit officer.\n" + _RULES +
    "\nWrite 3-5 plain sentences. Output prose only.")

_TYPED_SYS = (
    "You are a credit analyst writing a short explanation of a model's output "
    "for a credit officer.\n" + _RULES +
    "\nEvery claim you make must cite the evidence ids it rests on.\n"
    'Output strict JSON: {"claims":[{"text":"<one sentence>",'
    '"evidence_ids":["e1","e2"],"concept":"<one allowed concept or null>"}]}\n'
    f"Allowed concepts: {list(REASON_CONCEPTS)}")


def render_template_v2(eo: EvidenceObject, max_drivers: int = 4) -> Candidate:
    """Deterministic baseline on evidence/2.0: source names, documented labels,
    units and periods; no invention, no trend, no causal claim."""
    from evidence_builder import _fmt_value
    attrs = sorted(eo.by_type(EvidenceType.MODEL_ATTRIBUTION),
                   key=lambda a: -abs(a.contribution or 0.0))[:max_drivers]
    pred = eo.prediction
    unknown = eo.by_type(EvidenceType.UNSUPPORTED)
    facts = {f.eid: f for f in eo.by_type(EvidenceType.OBSERVED_FACT)}
    parts: List[str] = []
    claims: List[dict] = []
    if pred is not None:
        t = (f"The model's calibrated probability of default is {pred.calibrated_pd:.1%} "
             f"(outcome: {pred.target_definition}; horizon: {pred.horizon}).")
        parts.append(t)
        claims.append({"text": t, "evidence_ids": [pred.eid]})
    for a in attrs:
        f = facts.get((a.components or [None])[0])
        val = ""
        if f is not None:
            if f.value is None:
                val = f" ({f.value_label or 'value not recorded'})"
            else:
                # short unit only: the long unit text (ranges, caveats) stays in
                # the evidence, not in the explanation
                unit = re.split(r"\s*[(;]", f.unit or "")[0].strip()
                lab = f" ({f.value_label})" if f.value_label else (f" {unit}" if unit else "")
                when = (f" {f.observed_at}" if f.observed_at and
                        f.observed_at.startswith(("in ", "at ", "across ")) else "")
                val = f" (recorded value {_fmt_value(f.value)}{lab}{when})"
        verb = "raises" if a.direction == "increases_risk" else "lowers"
        t = f"{a.feature_group.replace(' (derived)', '')}{val} {verb} the model's risk score."
        parts.append(t)
        claims.append({"text": t, "evidence_ids": [a.eid] + ([f.eid] if f else [])})
    if attrs:
        t = (f"These contributions are on the {attrs[0].explained_scale or 'model score scale'}; "
             "they are not changes in the probability itself.")
        parts.append(t)
        claims.append({"text": t, "evidence_ids": [a.eid for a in attrs]})
    if unknown:
        t = ("No supported estimate is available for: "
             + "; ".join(u.reason or "unspecified item" for u in unknown) + ".")
        parts.append(t)
        claims.append({"text": t, "evidence_ids": [u.eid for u in unknown]})
    t = "This describes the model's behaviour, not a verified cause of default."
    parts.append(t)
    claims.append({"text": t, "evidence_ids": [pred.eid] if pred is not None else []})
    return Candidate(borrower_id=eo.borrower_id, arm="template", text=" ".join(parts),
                     claims=claims)


def _evidence_block(eo: EvidenceObject) -> str:
    lines = []
    for i in eo.items:
        d = i.to_dict()
        d.pop("eid", None)
        for k in ("aliases", "internal_name"):     # checker-only fields
            d.pop(k, None)
        lines.append(f"[{i.eid}] ({i.etype.value}) {json.dumps(d, default=str)}\n"
                     f"      permitted: {i.permitted()}")
    return "\n".join(lines)


def _cue_block(cue: Optional[str]) -> str:
    if not cue:
        return ""
    return ("\nEXPERIMENTAL CUE (injected by the study, NOT a verified borrower "
            f"attribute and NOT used by the predictor): {cue}\n")


# ---------------------------------------------------------------------------
# 3. LLM arms
# ---------------------------------------------------------------------------
def generate(eo: EvidenceObject, arm: str, family: str = "",
             *, cue: Optional[str] = None, temperature: float = 0.2,
             seed: Optional[int] = None, repetition: int = 0,
             experiment: str = "", prompt_version: int = 1) -> Candidate:
    if arm == "template":
        c = render_template_v2(eo) if prompt_version >= 2 else render_template(eo)
        c.cue, c.repetition = cue, repetition
        return c
    if not family:
        raise ValueError("an LLM arm needs a family")

    if prompt_version >= 2:
        sys_p = {"grounded": _GROUNDED_SYS_V2, "applicant": _APPLICANT_SYS_V2}.get(arm, _TYPED_SYS_V2)
    else:
        sys_p = _GROUNDED_SYS if arm == "grounded" else _TYPED_SYS
    user = (f"Borrower {eo.borrower_id} (dataset {eo.dataset}), prediction time "
            f"{eo.prediction_time}.{_cue_block(cue)}\nEVIDENCE:\n{_evidence_block(eo)}")

    try:
        r: LLMResult = GATEWAY.complete(
            family, [{"role": "system", "content": sys_p},
                     {"role": "user", "content": user}],
            # v2: 1500 tokens (700 truncated 14 of ~60 typed outputs in the first
            # M1 attempt); a truncated response is recorded as a failure, never repaired
            temperature=temperature, max_tokens=(1500 if prompt_version >= 2 else 700),
            want_json=(arm == "typed" or prompt_version >= 2), seed=seed,
            role=f"producer[{arm}]", experiment=experiment,
            item_id=(f"{eo.borrower_id}|{arm}|{family}" if prompt_version >= 2 else eo.borrower_id))
    except Exception as e:  # noqa: BLE001 - a failed generation is an observation
        return Candidate(borrower_id=eo.borrower_id, arm=arm, text="",
                         family=family, cue=cue, repetition=repetition,
                         error=f"{type(e).__name__}: {e}")

    if prompt_version >= 2:
        c = _finish_v2(eo, arm, r, cue, repetition, experiment)
        c.seed, c.temperature = seed, temperature
        return c

    claims: List[dict] = []
    text = r.text
    if arm == "typed":
        parsed = _parse_claims(r.text)
        if parsed is None:
            return Candidate(borrower_id=eo.borrower_id, arm=arm, text="",
                             family=family, model_id=r.model_id,
                             provider=r.provider, route_index=r.route_index,
                             latency_ms=r.latency_ms, cue=cue,
                             repetition=repetition, raw=r.text,
                             error="unparseable_claims_json")
        claims = parsed
        text = " ".join(c.get("text", "") for c in claims).strip()

    return Candidate(borrower_id=eo.borrower_id, arm=arm, text=text,
                     claims=claims, family=r.family, model_id=r.model_id,
                     provider=r.provider, route_index=r.route_index,
                     latency_ms=r.latency_ms, prompt_tokens=r.prompt_tokens,
                     completion_tokens=r.completion_tokens, cue=cue,
                     repetition=repetition, raw=r.text)


def _finish_v2(eo, arm, r, cue, repetition, experiment) -> Candidate:
    """producer/2.x: every response is JSON; unparseable -> parser judge."""
    import json_parse as JP
    schema, valid = ((JP.TYPED_SCHEMA, JP.valid_typed) if arm == "typed"
                     else (JP.GROUNDED_SCHEMA, JP.valid_grounded))
    obj, info = JP.parse_with_parser_judge(
        r.text, schema_hint=schema, validate=valid, experiment=experiment,
        item_id=f"{eo.borrower_id}|{arm}|{r.family}", role=f"producer[{arm}]",
        kind="producer", finish_reason=r.finish_reason)
    info["finish_reason"] = r.finish_reason
    base = dict(borrower_id=eo.borrower_id, arm=arm, family=r.family, model_id=r.model_id,
                provider=r.provider, route_index=r.route_index, latency_ms=r.latency_ms,
                prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                cue=cue, repetition=repetition, raw=r.text,
                parse_method=info["method"], parse_info=info,
                call_id=r.call_id, finish_reason=r.finish_reason)
    if obj is None:
        # error prefix = failure class: truncated_/filtered_/unparseable_ are
        # properties of the producer (kept); parser_unavailable_ is an outage
        tag = {"truncated": "truncated", "filtered": "filtered", "ambiguous": "ambiguous",
               "parser_unavailable": "parser_unavailable"}.get(info["method"], "unparseable")
        return Candidate(text="", error=f"{tag}_{arm}_json: {info.get('error', '')}"[:200], **base)
    if arm == "typed":
        claims = [{"text": str(c.get("text", "")).strip(),
                   "evidence_ids": [str(x) for x in (c.get("evidence_ids") or [])],
                   "concept": c.get("concept")} for c in obj["claims"] if isinstance(c, dict)]
        return Candidate(text=" ".join(c["text"] for c in claims).strip(), claims=claims, **base)
    return Candidate(text=obj["explanation"].strip(), **base)


def reparse(rec: dict, eo: EvidenceObject, experiment: str = "") -> Candidate:
    """Re-run ONLY the parsing contract on a stored producer response (used when
    the parser was unavailable): the producer is never re-sampled."""
    from types import SimpleNamespace
    r = SimpleNamespace(text=rec.get("raw") or "", family=rec["family"], model_id=rec.get("model_id"),
                        provider=rec.get("provider"), route_index=rec.get("route_index", 0),
                        latency_ms=rec.get("latency_ms", 0), prompt_tokens=rec.get("prompt_tokens", 0),
                        completion_tokens=rec.get("completion_tokens", 0), call_id=rec.get("call_id", ""),
                        finish_reason=rec.get("finish_reason"))
    c = _finish_v2(eo, rec["arm"], r, None, 0, experiment)
    c.seed, c.temperature = rec.get("seed"), rec.get("temperature")
    return c


def _parse_claims(raw: str) -> Optional[List[dict]]:
    """Tolerant JSON extraction. Returns None if no claim list survives."""
    t = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
    t = re.sub(r"\s*```$", "", t)
    for attempt in (t, _first_object(t)):
        if not attempt:
            continue
        try:
            d = json.loads(attempt)
        except Exception:
            try:
                from json_repair import repair_json
                d = json.loads(repair_json(attempt))
            except Exception:
                continue
        cl = d.get("claims") if isinstance(d, dict) else None
        if isinstance(cl, list) and cl:
            out = []
            for c in cl:
                if not isinstance(c, dict):
                    continue
                out.append({"text": str(c.get("text", "")).strip(),
                            "evidence_ids": [str(x) for x in
                                             (c.get("evidence_ids") or [])],
                            "concept": c.get("concept")})
            if out:
                return out
    return None


def _first_object(text: str) -> Optional[str]:
    s = text.find("{")
    if s == -1:
        return None
    depth, instr, esc = 0, False, False
    for i in range(s, len(text)):
        ch = text[i]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        elif ch == '"':
            instr = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[s:i + 1]
    return None


if __name__ == "__main__":
    from evidence import EvidenceItem, new_item
    eo = EvidenceObject(borrower_id="DEMO-1", dataset="sba",
                        prediction_time="approval_date")
    eo.add(new_item("f1", EvidenceType.OBSERVED_FACT, source_field="GrAppv",
                    value=455000.0, unit="USD", observed_at="approval_date"))
    eo.add(new_item("a1", EvidenceType.MODEL_ATTRIBUTION, model_id="m0",
                    feature_group="GrAppv", contribution=0.41,
                    direction="increases_risk",
                    explained_scale="raw log-odds margin",
                    reference_background="training-fold mean"))
    eo.add(new_item("p1", EvidenceType.MODEL_PREDICTION, calibrated_pd=0.31,
                    target_definition="MIS_Status == CHGOFF",
                    horizon="loan lifetime", model_id="m0",
                    calibrator_id="iso0"))
    print("── template ──\n", render_template(eo).text, "\n")
    c = generate(eo, "typed", "llama", experiment="selftest")
    print(f"── typed via {c.family}:{c.model_id} ── err={c.error}")
    for cl in c.claims:
        print(f"   • {cl['text']}  <- {cl['evidence_ids']}")
