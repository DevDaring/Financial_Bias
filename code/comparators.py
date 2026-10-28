"""
Comparator verifiers for E1/E3 (plan §6): reimplementations of the closest
recent work, adapted to binary credit-explanation verification. Every LLM-based
comparator takes the judge model as an argument, so a comparison can hold the
model fixed and vary only the method (plan §8: method vs model effects).

  always_accept / always_reject     diagnostic controls (plan E1)
  generic_judge                     repaired evidence-grounded judge (judge/2.0 in verify.py)
  pt_audit                          Pratama & Tseng (2026), arXiv:2608.08126: deterministic
                                    Membership / Direction / Coverage audit of a narrative
  zandi_judge                       Zandi et al. (2026), arXiv:2608.17715: LLM-as-judge
                                    Tabular Feature Coverage (TFC) and Tabular Directional
                                    Consistency (TDC), 1-5 scales
  mermigkis_verifier / refeed       Mermigkis et al. (WCCI 2026), arXiv:2604.12543: Verifier
                                    LLM (verdict, confidence, justification, one of six error
                                    types) and the refeed loop; prompts adapted from the authors'
                                    public templates (github.com/gMerm/Two-Stage-LLM-Framework-
                                    For-Verified-XAI-Explanations)
  pbp_judge                         Tapwal et al. (2026), arXiv:2605.23970: Proof-Before-
                                    Preference (evidence lock, then a verdict from the locked
                                    notes only), adapted from ranking to a binary verdict
  generic_tool_judge  (B2)          the repaired generic judge with the SAME evidence lookup and
                                    arithmetic tools the proposed method uses internally (lookup,
                                    find, compare_value, direction, strongest, dated_values), at the
                                    same call budget (Findeis et al., ACL 2025: tools help judges)
  generic_second_look               cost-matched control: the generic judge, then one more call that
                                    re-checks its own verdict (an extra call alone must not explain a gain)
  WeakStrongPolicy                  Kiyani et al. (ICML 2026), arXiv:2602.17633: accept /
                                    reject / defer-to-strong with two online-calibrated
                                    thresholds; update rule ported from the authors' Apache-2.0
                                    code (github.com/nooranisima/weak-and-strong-verification,
                                    src/algorithms/adaptive.py)
  select_release_policy             pooled vs group-audited policy selection with simultaneous
                                    Clopper-Pearson bounds (Learn-then-Test style; plan §4.4)

Departures from each original are stated in DEPARTURES and must be reported in
the paper.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import contract as C
import gateway as G
import json_parse as JP
import verify as V
from evidence import EvidenceObject, EvidenceType
from generate import Candidate

DEPARTURES = {
    "pt_audit": "Original audits a 2.7B model's narrative against the top-4 fused SHAP/LIME drivers; here "
                "the driver set is the evidence's attributions (4 per borrower) and feature mentions are "
                "matched with the study's documented field names and aliases.",
    "zandi_judge": "Original judge was Claude Sonnet 4.6 scoring TFC/TDC (and network metrics) on 1-5; here "
                   "only the tabular metrics apply, the judge model is held fixed across methods, and a "
                   "release threshold (both scores >= tfc_min/tdc_min, default 4) is added and must be "
                   "chosen on calibration data.",
    "mermigkis_verifier": "Original verifies an explanation plus its chain of thought; our explanations have no "
                          "CoT, so the CoT section and the MissingCoT error type are dropped. Dataset context "
                          "is the evidence object. Error types otherwise verbatim.",
    "mermigkis_refeed": "Original allows up to K=10 refeed iterations; here K is a budget parameter (default 2).",
    "pbp_judge": "Original locks criterion-wise notes, scores 0-5, then ranks summaries; here notes are locked "
                 "for the contract criteria, and turn 2 gives a binary verdict whose violations must quote "
                 "text that appears in the locked notes.",
    "generic_tool_judge": "Findeis et al. give judges web search / code execution / math tools on general "
                          "tasks; here the tools are the evidence-lookup and arithmetic primitives of the "
                          "proposed method (not its typed claim execution), offered in one tool round "
                          "before the verdict so the call budget equals the proposed method's.",
    "weak_strong": "Original strong verifier is ground truth / human inspection on MATH and Sudoku; here the "
                   "strong verifier is human review (the annotation label), and the weak score is a cheap "
                   "judge's confidence that the explanation is correct.",
}


# ---------------------------------------------------------------------------
# common result
# ---------------------------------------------------------------------------
@dataclass
class CVerdict:
    method: str
    accepted: Optional[bool]                 # None = unresolved
    judge_family: Optional[str] = None
    score: Optional[float] = None            # a continuous signal when the method has one
    detail: dict = field(default_factory=dict)
    error: Optional[str] = None
    calls: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def always_accept(cand: Candidate, eo: EvidenceObject, **_) -> CVerdict:
    return CVerdict("always_accept", True)


def always_reject(cand: Candidate, eo: EvidenceObject, **_) -> CVerdict:
    return CVerdict("always_reject", False)


def _llm(cand: Candidate, family: str, messages: List[dict], *, schema: str, validate: Callable,
         verdict_key: Optional[str], max_tokens: int, experiment: str, role: str):
    item = f"{cand.borrower_id}|{cand.arm}|{cand.family}"
    gw = G.GATEWAY
    if cand.family not in ("template", ""):
        r = gw.judged_by(family, cand.family, messages, temperature=0.0, max_tokens=max_tokens,
                         want_json=True, experiment=experiment, item_id=item, role=role)
    else:
        r = gw.complete(family, messages, temperature=0.0, max_tokens=max_tokens, want_json=True,
                        experiment=experiment, item_id=item, role=role)
    call = {"family": family, "model_id": r.model_id, "provider": r.provider, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens, "latency_ms": r.latency_ms,
            "finish_reason": r.finish_reason}
    d, info = JP.parse_with_parser_judge(r.text, schema_hint=schema, validate=validate, verdict_key=verdict_key,
                                         experiment=experiment, item_id=item, role=role, kind="comparator",
                                         finish_reason=r.finish_reason)
    call["parse"] = info.get("method")
    return d, call, r.text


def _safe(method: str, family: str, fn) -> CVerdict:
    try:
        return fn()
    except G.JudgeIndependenceError as e:
        return CVerdict(method, None, family, error=f"config: {e}")
    except Exception as e:  # noqa: BLE001 - an outage is an unresolved verdict
        return CVerdict(method, None, family, error=f"transport: {type(e).__name__}: {str(e)[:120]}")


# ---------------------------------------------------------------------------
# repaired generic judge (judge/2.0) - wraps verify.semantic_check
# ---------------------------------------------------------------------------
def generic_judge(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "", **_) -> CVerdict:
    v = V.semantic_check(cand, eo, family, experiment=experiment, prompt_version=2)
    return CVerdict("generic_judge2", None if v.error else v.accepted, family,
                    detail={"violations": v.violations, "parse": v.parse_method}, error=v.error,
                    calls=[{"family": family, "model_id": v.judge_model_id, "provider": v.provider,
                            "prompt_tokens": v.prompt_tokens, "completion_tokens": v.completion_tokens,
                            "latency_ms": v.latency_ms}])


# ---------------------------------------------------------------------------
# Pratama & Tseng (2026): Membership / Direction / Coverage
# ---------------------------------------------------------------------------
_UPDOWN = re.compile(r"\b(?:" + V._UP_INFL + "|" + V._UP_BARE + r"|higher risk|adds? to (?:the )?risk)\b", re.I)
_DOWNUP = re.compile(r"\b(?:" + V._DN_INFL + "|" + V._DN_BARE + r"|lower risk|offsets?)\b", re.I)


def _dataset_vocab(eo: EvidenceObject) -> Dict[str, List[str]]:
    """Every documented field of the dataset: name -> names/aliases (for Membership)."""
    try:
        import feature_semantics as FS
        sp = FS.specs(eo.dataset)
    except Exception:  # noqa: BLE001
        return {}
    return {k: [k] + list(getattr(s, "aliases", []) or []) for k, s in sp.items()}


def pt_audit(cand: Candidate, eo: EvidenceObject, **_) -> CVerdict:
    attrs = eo.by_type(EvidenceType.MODEL_ATTRIBUTION)
    names = {a.eid: C._item_names(eo, a.eid) for a in attrs}
    supplied_fields = set()
    for a in attrs:
        supplied_fields.add(str(a.feature_group))
        for c in a.components or []:
            it = eo.get(c)
            if it is not None and it.source_field:
                supplied_fields.add(str(it.source_field).replace(" (derived)", ""))
    low = V._canon(cand.text.lower())
    problems = []
    # Membership: every feature named must be in the supplied driver set
    for fld, al in _dataset_vocab(eo).items():
        if fld in supplied_fields or not al:
            continue
        if V._mentions(low, [x for x in al if len(x) > 3]):
            problems.append(("membership", fld))
    # Direction: for each named driver, the asserted direction must match its sign
    for sent in re.split(r"(?<=[.;!?])\s+", cand.text):
        sl = V._canon(sent.lower())
        named = [e for e, ns in names.items() if ns and V._mentions(sl, ns)]
        if len(named) != 1:
            continue                         # direction of a multi-feature sentence is not attributable
        a = eo.get(named[0])
        up, dn = bool(_UPDOWN.search(sent)), bool(_DOWNUP.search(sent))
        if up == dn or V._governed_neg(sent[: (_UPDOWN.search(sent) or _DOWNUP.search(sent)).start()]):
            continue
        if (a.direction == "increases_risk") != up:
            problems.append(("direction", a.feature_group))
    # Coverage: the top driver by |a| must be named
    if attrs:
        top = max(attrs, key=lambda x: abs(x.contribution or 0.0))
        if not V._mentions(low, names[top.eid]):
            problems.append(("coverage", top.feature_group))
    return CVerdict("pt_audit", not problems, detail={"failures": problems})


# ---------------------------------------------------------------------------
# Zandi et al. (2026): TFC and TDC, 1-5
# ---------------------------------------------------------------------------
ZANDI_SYS = (
    "You evaluate an explanation of a credit-risk model's prediction against the model evidence. "
    "Score two metrics on a 1-5 scale (5 = best):\n"
    "Tabular Feature Coverage (TFC): measures whether the explanation mentions the important tabular "
    "features that the predictive model actually used.\n"
    "Tabular Directional Consistency (TDC): measures whether the explanation correctly describes the "
    "direction of the effect of each tabular feature.\n"
    'Reply with JSON only: {"TFC": <1-5>, "TDC": <1-5>, "rationale": "<one or two sentences>"}')
ZANDI_SCHEMA = '{"TFC": 1, "TDC": 1, "rationale": "<text>"}'


def _valid_zandi(o) -> bool:
    return isinstance(o, dict) and all(isinstance(o.get(k), (int, float)) and 1 <= o[k] <= 5 for k in ("TFC", "TDC"))


def zandi_judge(cand: Candidate, eo: EvidenceObject, family: str, *, tfc_min: float = 4, tdc_min: float = 4,
                experiment: str = "", **_) -> CVerdict:
    def run():
        msgs = [{"role": "system", "content": ZANDI_SYS},
                {"role": "user", "content": json.dumps({"evidence": eo.for_generator(), "explanation": cand.text},
                                                       default=str)[:14000]}]
        d, call, raw = _llm(cand, family, msgs, schema=ZANDI_SCHEMA, validate=_valid_zandi, verdict_key=None,
                            max_tokens=600, experiment=experiment, role=f"zandi[{family}]")
        if d is None:
            return CVerdict("zandi_judge", None, family, error=f"unparseable: {call['parse']}", calls=[call])
        tfc, tdc = float(d["TFC"]), float(d["TDC"])
        return CVerdict("zandi_judge", tfc >= tfc_min and tdc >= tdc_min, family, score=(tfc + tdc) / 10,
                        detail={"TFC": tfc, "TDC": tdc, "rationale": str(d.get("rationale", ""))[:300],
                                "thresholds": [tfc_min, tdc_min]}, calls=[call])
    return _safe("zandi_judge", family, run)


# ---------------------------------------------------------------------------
# Mermigkis et al. (WCCI 2026): Verifier LLM and refeed
# ---------------------------------------------------------------------------
MERMIGKIS_ERROR_TYPES = ["SwapTopFeature", "SwapMinorFeature", "NegateRelation", "OmitFeature",
                         "InsertHallucination", "None"]          # MissingCoT dropped: no CoT here
MERMIGKIS_SYS = """## Role
You are a Verifier model specialized in evaluating the faithfulness, logical soundness, and factual consistency of LLM-generated explanations for XAI method outputs.
You must carefully inspect the raw XAI data and the final explanation to ensure both are aligned and accurate. You MUST answer strictly according to the "How-to-reply" guide.

## Guidelines
- You must act as an evaluator-verifier: do not produce new explanations.
- All judgments must be supported by explicit evidence found in the input sections.
- The explanation that you'll receive for judgement is the output of another LLM which receives information about the dataset and its features.

## Problem Statement
You are tasked with evaluating a verbal explanation produced by an LLM (the "Explainer") that attempts to interpret the output of an XAI method (SHAP attributions of a credit-risk model). Your goal is to determine whether the explanation is faithful, logically consistent, and non-hallucinated relative to the provided XAI data.

## Solution Structure
1. Evaluate the explanation for clarity and impartiality, avoiding subjective or emotional language.
2. Confirm there is not swapping in the Explanation for each feature. Flag these either as SwapTopFeature or SwapMinorFeature error.
3. Ensure that the directional relationship described for each feature is logically consistent and not reversed in the Explanation. Flag this as NegateRelation error.
4. Ensure that all explained features are relevant to the answer and no feature is omitted or forgotten in the ranking. Flag this as OmitFeature error, except in the case of local explanations, where omitting features is allowed.
5. Flag as InsertHallucination only if the explanation introduces:
   - Causal claims that contradict the XAI data
   - Quantitative assertions (thresholds, percentages) that are fabricated or unsupported
   - Contradictions between the XAI data and stated mechanisms
   - Factually false statements about the data or model

   **Do not flag as hallucination** if the explanation uses reasonable causal or directional phrasing that is consistent with the XAI values or standard domain understanding.

6. Mark as InsertHallucination only if an assumption cannot be logically supported by either the XAI method output **or** by well-established domain reasoning relevant to the dataset.

### How to reply
Respond only with the requested JSON structure. Place your verification judgement in the following fields:
{
    "confidence": float between 0 and 1, representing your certainty in the evaluation,
    "verdict": "accept" or "reject", indicating whether the explanation meets the verification criteria,
    "justification": brief, explicit reasoning referencing evidence from the XAI Content and Explanation sections,
    "error type": single phrase from ["SwapTopFeature", "SwapMinorFeature", "NegateRelation", "OmitFeature", "InsertHallucination", "None"]
}"""
MERMIGKIS_SCHEMA = ('{"confidence": 0.9, "verdict": "accept|reject", "justification": "<text>", '
                    '"error type": "<one of the six>"}')


def _valid_mermigkis(o) -> bool:
    return (isinstance(o, dict) and str(o.get("verdict", "")).lower() in ("accept", "reject")
            and isinstance(o.get("confidence"), (int, float)) and 0 <= float(o["confidence"]) <= 1)


def mermigkis_verifier(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "", **_) -> CVerdict:
    def run():
        user = f"## XAI Content\n{json.dumps(eo.for_generator(), default=str)[:12000]}\n\n## Explanation\n{cand.text}"
        msgs = [{"role": "system", "content": MERMIGKIS_SYS}, {"role": "user", "content": user}]
        d, call, raw = _llm(cand, family, msgs, schema=MERMIGKIS_SCHEMA, validate=_valid_mermigkis,
                            verdict_key=None, max_tokens=700, experiment=experiment, role=f"mermigkis[{family}]")
        if d is None:
            return CVerdict("mermigkis_verifier", None, family, error=f"unparseable: {call['parse']}", calls=[call])
        acc = str(d["verdict"]).lower() == "accept"
        conf = float(d["confidence"])
        return CVerdict("mermigkis_verifier", acc, family,
                        score=conf if acc else 1.0 - conf,            # P(explanation is correct)
                        detail={"error_type": d.get("error type") or d.get("error_type"),
                                "justification": str(d.get("justification", ""))[:400], "confidence": conf},
                        calls=[call])
    return _safe("mermigkis_verifier", family, run)


MERMIGKIS_REFEED_USER = (
    "Previously, the explanation given by an LLM was flagged as erroneous with the justification under "
    "\"Verifier Justification\". Please provide a CORRECTED explanation that addresses the verifier's concerns. "
    "Base your explanation ONLY on the provided evidence. Do not invent details.\n\n"
    "## XAI Output\n{xai}\n\n## Previous Explanation\n{prev}\n\n## Verifier Justification\n{just}\n\n"
    'Reply with JSON only: {{"explanation": "<the corrected explanation>"}}')


def mermigkis_refeed(cand: Candidate, eo: EvidenceObject, producer_family: str, verifier_family: str, *,
                     max_rounds: int = 2, experiment: str = "") -> Tuple[Candidate, List[CVerdict]]:
    """Explainer -> Verifier -> refeed, at most max_rounds corrections (E3 generation comparator)."""
    history: List[CVerdict] = []
    cur = cand
    for k in range(max_rounds + 1):
        v = mermigkis_verifier(cur, eo, verifier_family, experiment=experiment)
        history.append(v)
        if v.accepted is not False or k == max_rounds:
            break
        msgs = [{"role": "user", "content": MERMIGKIS_REFEED_USER.format(
            xai=json.dumps(eo.for_generator(), default=str)[:12000], prev=cur.text,
            just=v.detail.get("justification", ""))}]
        r = G.GATEWAY.complete(producer_family, msgs, temperature=0.0, max_tokens=1200, want_json=True,
                               experiment=experiment, item_id=f"{cand.borrower_id}|refeed{k + 1}",
                               role=f"refeed[{producer_family}]")
        d, _ = JP.parse_with_parser_judge(r.text, schema_hint=JP.GROUNDED_SCHEMA, validate=JP.valid_grounded,
                                          experiment=experiment, item_id=f"{cand.borrower_id}|refeed{k + 1}",
                                          role="refeed", kind="producer", finish_reason=r.finish_reason)
        if d is None:
            break
        cur = Candidate(borrower_id=cand.borrower_id, arm=f"{cand.arm}+refeed{k + 1}", text=d["explanation"],
                        family=producer_family, model_id=r.model_id, provider=r.provider)
    return cur, history


# ---------------------------------------------------------------------------
# Tapwal et al. (2026): Proof-Before-Preference, binary adaptation
# ---------------------------------------------------------------------------
PBP_LOCK = (
    "You will assess a credit-risk explanation against the model evidence, in two turns.\n"
    "TURN 1 - EVIDENCE LOCK. Write criterion-wise notes for these criteria: (a) values and units, "
    "(b) direction of each attribution, (c) claims about change over time or causes, (d) which drivers "
    "and the probability estimate are conveyed. For every note, quote the exact span(s) from the Evidence "
    "or the Explanation that support it. Do NOT give any verdict yet.\n"
    'Reply with JSON only: {"notes":[{"criterion":"a|b|c|d","note":"...","quotes":["<exact span>"]}]}')
PBP_VERDICT = (
    "TURN 2 - VERDICT. Your notes are now locked. Using ONLY the locked notes (do not add new evidence), "
    "decide whether the explanation contains a material error. Every violation must quote a span that "
    "appears in your locked notes.\n"
    'Reply with JSON only: {"accepted": true|false, "violations":[{"type":"<type>","detail":"...",'
    '"quote":"<span from the locked notes>"}]}')
PBP_LOCK_SCHEMA = '{"notes": [{"criterion": "a", "note": "<text>", "quotes": ["<span>"]}]}'


def _valid_notes(o) -> bool:
    return isinstance(o, dict) and isinstance(o.get("notes"), list) and all(
        isinstance(n, dict) and isinstance(n.get("quotes", []), list) for n in o["notes"])


def pbp_judge(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "", **_) -> CVerdict:
    def run():
        user = json.dumps({"evidence": eo.for_generator(), "explanation": cand.text}, default=str)[:14000]
        m1 = [{"role": "system", "content": PBP_LOCK}, {"role": "user", "content": user}]
        notes, c1, raw1 = _llm(cand, family, m1, schema=PBP_LOCK_SCHEMA, validate=_valid_notes, verdict_key=None,
                               max_tokens=1500, experiment=experiment, role=f"pbp-lock[{family}]")
        if notes is None:
            return CVerdict("pbp_judge", None, family, error=f"lock unparseable: {c1['parse']}", calls=[c1])
        m2 = m1 + [{"role": "assistant", "content": json.dumps(notes)},
                   {"role": "user", "content": PBP_VERDICT}]
        d, c2, raw2 = _llm(cand, family, m2, schema=JP.JUDGE_SCHEMA, validate=JP.valid_judge,
                           verdict_key="accepted", max_tokens=800, experiment=experiment,
                           role=f"pbp-verdict[{family}]")
        if d is None:
            return CVerdict("pbp_judge", None, family, error=f"verdict unparseable: {c2['parse']}", calls=[c1, c2])
        locked = " ".join(q for n in notes["notes"] for q in (n.get("quotes") or []) if isinstance(q, str))
        viol = [v for v in d.get("violations") or [] if isinstance(v, dict)]
        unlocked = [v for v in viol if v.get("quote") and C.quote_position(str(v["quote"]), locked) < 0]
        if d["accepted"] != (not viol) or unlocked:
            return CVerdict("pbp_judge", None, family, calls=[c1, c2],
                            error=("violation not grounded in the locked notes" if unlocked
                                   else "verdict and violations disagree"),
                            detail={"violations": viol, "n_notes": len(notes["notes"])})
        return CVerdict("pbp_judge", d["accepted"], family, detail={"violations": viol, "n_notes": len(notes["notes"])},
                        calls=[c1, c2])
    return _safe("pbp_judge", family, run)


# ---------------------------------------------------------------------------
# Kiyani et al. (ICML 2026): weak-strong verification with online thresholds
# ---------------------------------------------------------------------------
class WeakStrongPolicy:
    """Two-threshold accept / reject / defer policy with the authors' online updates.

    Ported from src/algorithms/adaptive.py (SimulatedAdaptiveRunner.update_thresholds),
    Apache License 2.0, Copyright the authors. Weak score w in [0,1] = estimated
    probability the explanation is correct; the strong verifier H is human review.
    """

    def __init__(self, alpha: float = 0.10, beta: float = 0.10, eta: float = 0.1, eta_r: float = 0.1,
                 p_a_init: float = 0.3, p_r_init: float = 0.3, p_a_min: float = 0.05, p_r_min: float = 0.05,
                 tau_a: Optional[float] = None, tau_r: Optional[float] = None, seed: int = 0):
        import random
        self.alpha, self.beta, self.eta, self.eta_r = alpha, beta, eta, eta_r
        self.p_a_init, self.p_r_init, self.p_a_min, self.p_r_min = p_a_init, p_r_init, p_a_min, p_r_min
        self.tau_a = 1 - alpha if tau_a is None else tau_a
        self.tau_r = beta if tau_r is None else tau_r
        self.t = 0
        self.rng = random.Random(seed)

    def _p_a(self) -> float:
        return max(self.p_a_min, self.p_a_init / math.sqrt(self.t + 1))

    def _p_r(self) -> float:
        return max(self.p_r_min, self.p_r_init / math.sqrt(self.t + 1))

    def _update(self, w: float, h: bool) -> None:
        g = 0.0
        if w > self.tau_a:
            g = ((not h) * (1 - self.alpha)) / self._p_a()
        elif w > self.tau_r:
            g = (not h) * (-self.alpha)
        else:
            g = ((not h) * (-self.alpha)) / self._p_r()
        self.tau_a = max(self.tau_r + 0.05, min(1.0, self.tau_a + self.eta * g))
        gb = 0.0
        if w > self.tau_a:
            gb = (h * self.beta) / self._p_a()
        elif w > self.tau_r:
            gb = h * self.beta
        else:
            gb = (h * (-(1 - self.beta))) / self._p_r()
        self.tau_r = max(0.0, min(self.tau_a - 0.05, self.tau_r + self.eta_r * gb))

    def step(self, w: float, strong: Callable[[], bool]) -> dict:
        """Decide one item. `strong()` is only called when the policy queries it."""
        self.t += 1
        if w > self.tau_a:
            region, decision = "accept", "accept"
            queried = self.rng.random() < self._p_a()
        elif w > self.tau_r:
            region, queried, decision = "defer", True, None
        else:
            region, decision = "reject", "reject"
            queried = self.rng.random() < self._p_r()
        h = strong() if queried else None
        if region == "defer":
            decision = "accept" if h else "reject"
        if queried:
            self._update(w, bool(h))
        return {"region": region, "decision": decision, "queried": queried,
                "tau_a": round(self.tau_a, 4), "tau_r": round(self.tau_r, 4)}


def run_weak_strong(stream: Sequence[Tuple[float, bool]], **kw) -> dict:
    """Offline evaluation on (weak score, true correctness) pairs, in stream order."""
    pol = WeakStrongPolicy(**kw)
    rows = [dict(pol.step(w, lambda h=h: h), truth=h) for w, h in stream]
    acc_wrong = sum(1 for r in rows if r["decision"] == "accept" and not r["truth"])
    rej_right = sum(1 for r in rows if r["decision"] == "reject" and r["truth"])
    n_wrong = sum(1 for r in rows if not r["truth"])
    n_right = len(rows) - n_wrong
    return {"n": len(rows), "strong_calls": sum(r["queried"] for r in rows),
            "strong_rate": round(sum(r["queried"] for r in rows) / max(1, len(rows)), 3),
            "type1_false_accept": round(acc_wrong / max(1, n_wrong), 3),
            "type2_false_reject": round(rej_right / max(1, n_right), 3),
            "final_tau_a": rows[-1]["tau_a"] if rows else None, "final_tau_r": rows[-1]["tau_r"] if rows else None}


# ---------------------------------------------------------------------------
# release-policy selection: pooled vs group-audited (plan §4.4)
# ---------------------------------------------------------------------------
def select_release_policy(items: List[dict], policies: Dict[str, Callable[[dict], bool]], *, eps: float,
                          delta: float = 0.05, group_audited: bool = True) -> dict:
    """items: {"group": str, "loss": bool, ...signals}. Choose the policy maximising the minimum
    group coverage whose (simultaneous, Bonferroni delta/(M*G)) Clopper-Pearson upper bound on
    selective error is <= eps, either per group (group-audited) or pooled. Exact counts only."""
    from policy import clopper_pearson_upper
    groups = sorted({it["group"] for it in items})
    m = len(policies)
    rows = []
    for name, rel in policies.items():
        per = {}
        for g in groups:
            gi = [it for it in items if it["group"] == g]
            r = [it for it in gi if rel(it)]
            k = sum(1 for it in r if it["loss"])
            per[g] = {"n": len(gi), "released": len(r), "k": k, "coverage": round(len(r) / max(1, len(gi)), 4)}
        if group_audited:
            alpha = delta / (m * len(groups))
            for g in groups:
                per[g]["ub"] = round(clopper_pearson_upper(per[g]["k"], per[g]["released"], alpha), 4)
            ok = all(per[g]["ub"] <= eps for g in groups)
        else:
            alpha = delta / m
            K = sum(p["k"] for p in per.values()); N = sum(p["released"] for p in per.values())
            ub = clopper_pearson_upper(K, N, alpha)
            for g in groups:
                per[g]["ub_pooled"] = round(ub, 4)
            ok = ub <= eps
        rows.append({"policy": name, "feasible": ok, "min_coverage": min(p["coverage"] for p in per.values()),
                     "per_group": per})
    feas = [r for r in rows if r["feasible"]]
    best = max(feas, key=lambda r: r["min_coverage"]) if feas else None
    return {"mode": "group_audited" if group_audited else "pooled", "eps": eps, "delta": delta,
            "chosen": best["policy"] if best else None, "rows": rows}


# ---------------------------------------------------------------------------
# B2: generic judge + the same lookup / arithmetic tools (one tool round)
# ---------------------------------------------------------------------------
TOOL_SPEC = (
    "\n\nBefore the verdict you MAY request tool results ONCE. To do so reply with JSON only: "
    '{"tool_calls":[{"tool":"<name>","args":{...}}]} (at most 8 calls). Tools:\n'
    "- lookup {evidence_id}: the full evidence item\n"
    "- find {name}: evidence ids whose documented names match the words\n"
    "- compare_value {evidence_id, number, percent}: does the number match the item's value "
    "(documented rounding, units and percent equivalents)?\n"
    "- direction {evidence_id}: the attribution's sign and contribution\n"
    "- strongest {direction?}: attribution(s) with the largest |contribution| (ties included), "
    "optionally among increases_risk or decreases_risk only\n"
    "- dated_values {evidence_id}: the dated observations of the item's series, oldest first\n"
    "You then receive the results and must reply with the verdict JSON. You may also reply with "
    "the verdict JSON immediately.")


def run_tool(eo: EvidenceObject, call: dict) -> dict:
    import witness as W
    t = str(call.get("tool") or call.get("name") or "")
    a = call.get("args") if isinstance(call.get("args"), dict) else {}
    try:
        if t == "lookup":
            it = eo.get(str(a.get("evidence_id")))
            return {"tool": t, "result": it.to_dict() if it else "unknown id"}
        if t == "find":
            ms = W.bind(str(a.get("name") or ""), eo)
            return {"tool": t, "result": [m.eid for m in ms]}
        if t == "compare_value":
            it = eo.get(str(a.get("evidence_id")))
            if it is None:
                return {"tool": t, "result": "unknown id"}
            num = float(str(a.get("number")).replace(",", "").rstrip("%"))
            pct = bool(a.get("percent")) or str(a.get("number")).strip().endswith("%")
            n = {"v": num, "tol": 0.5 * 10 ** -(len(str(num).split(".")[1]) if "." in str(num) else 0),
                 "pct": pct, "pp": False, "signed": False, "lo": num, "hi": num}
            vals = W._values(eo, it) if it.etype is not EvidenceType.MODEL_PREDICTION else [it.calibrated_pd]
            return {"tool": t, "result": {"matches": W._num_ok(n, vals, W._pct_values(it)), "evidence_values": vals[:6]}}
        if t == "direction":
            it = eo.get(str(a.get("evidence_id")))
            return {"tool": t, "result": ({"direction": it.direction, "contribution": it.contribution}
                                          if it is not None and it.etype is EvidenceType.MODEL_ATTRIBUTION
                                          else "not an attribution")}
        if t == "strongest":
            at = eo.by_type(EvidenceType.MODEL_ATTRIBUTION)
            d = a.get("direction")
            if d in ("increases_risk", "decreases_risk"):
                at = [x for x in at if x.direction == d]
            if not at:
                return {"tool": t, "result": []}
            top = max(abs(x.contribution or 0) for x in at)
            return {"tool": t, "result": [{"evidence_id": x.eid, "contribution": x.contribution} for x in at
                                          if abs(abs(x.contribution or 0) - top) <= W.TIE_TOL]}
        if t == "dated_values":
            ser = W._series_items(eo, str(a.get("evidence_id")))
            return {"tool": t, "result": [{"evidence_id": x.eid, "observed_at": x.observed_at, "value": x.value}
                                          for x in ser]}
    except Exception as e:  # noqa: BLE001 - a bad tool call is reported to the judge, not raised
        return {"tool": t, "error": f"{type(e).__name__}: {str(e)[:80]}"}
    return {"tool": t, "error": "unknown tool"}


def _tool_calls(text: str) -> Optional[List[dict]]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None
    tc = d.get("tool_calls") if isinstance(d, dict) else None
    return [c for c in tc if isinstance(c, dict)][:8] if isinstance(tc, list) else None


def generic_tool_judge(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "",
                       max_calls: int = 2, **_) -> CVerdict:
    sys_ = V.JUDGE_PROMPTS[2] + TOOL_SPEC
    payload = json.dumps({"evidence": eo.for_generator(), "explanation": cand.text}, default=str)[:12000]
    msgs = [{"role": "system", "content": sys_}, {"role": "user", "content": payload}]
    item = f"{cand.borrower_id}|{cand.arm}|{cand.family}"

    def run():
        calls, tools_used = [], []
        gw = G.GATEWAY
        for k in range(max_calls):
            kw = dict(temperature=0.0, max_tokens=1500, want_json=True, experiment=experiment,
                      item_id=item + (f"|t{k}" if k else ""), role=f"tooljudge[{family}]")
            r = (gw.judged_by(family, cand.family, msgs, **kw) if cand.family not in ("template", "")
                 else gw.complete(family, msgs, **kw))
            calls.append({"family": family, "model_id": r.model_id, "provider": r.provider,
                          "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                          "latency_ms": r.latency_ms, "finish_reason": r.finish_reason})
            tc = _tool_calls(r.text)
            if tc is not None and k < max_calls - 1:
                res = [run_tool(eo, c) for c in tc]
                tools_used += res
                msgs.extend([{"role": "assistant", "content": r.text},
                             {"role": "user", "content": "Tool results: " + json.dumps(res, default=str)[:6000]
                              + " Now reply with the verdict JSON only."}])
                continue
            d, info = JP.parse_with_parser_judge(r.text, schema_hint=JP.JUDGE_SCHEMA, validate=JP.valid_judge,
                                                 verdict_key="accepted", experiment=experiment, item_id=item,
                                                 role=f"tooljudge[{family}]", kind="comparator",
                                                 finish_reason=r.finish_reason)
            if d is None:
                return CVerdict("generic_tool_judge", None, family, error=f"unparseable: {info.get('method')}",
                                detail={"tools": tools_used}, calls=calls)
            viol = [x for x in (d.get("violations") or []) if isinstance(x, dict)]
            if d["accepted"] != (not viol):
                return CVerdict("generic_tool_judge", None, family, error="inconsistent_judge_output",
                                detail={"violations": viol, "tools": tools_used}, calls=calls)
            return CVerdict("generic_tool_judge", d["accepted"], family,
                            detail={"violations": viol, "tools": tools_used, "parse": info.get("method")},
                            calls=calls)
        return CVerdict("generic_tool_judge", None, family, error="no verdict within the call budget",
                        detail={"tools": tools_used}, calls=calls)
    return _safe("generic_tool_judge", family, run)


SECOND_LOOK_USER = ("Your previous verdict on this explanation was: {prev}. Re-check every violation against "
                    "the evidence values and directions; drop any the evidence does not show, add any you "
                    "missed. Reply with the final verdict JSON only.")


def generic_second_look(cand: Candidate, eo: EvidenceObject, family: str, *, experiment: str = "", **_) -> CVerdict:
    """Cost-matched control: generic judge + one self-review call."""
    first = generic_judge(cand, eo, family, experiment=experiment)
    if first.error:
        return CVerdict("generic_second_look", None, family, error=first.error, calls=first.calls)
    prev = json.dumps({"accepted": first.accepted, "violations": first.detail.get("violations", [])})[:3000]
    msgs = [{"role": "system", "content": V.JUDGE_PROMPTS[2]},
            {"role": "user", "content": json.dumps({"evidence": eo.for_generator(), "explanation": cand.text},
                                                   default=str)[:12000]},
            {"role": "user", "content": SECOND_LOOK_USER.format(prev=prev)}]

    def run():
        d, call, _raw = _llm(cand, family, msgs, schema=JP.JUDGE_SCHEMA, validate=JP.valid_judge,
                             verdict_key="accepted", max_tokens=1500, experiment=experiment,
                             role=f"secondlook[{family}]")
        if d is None:
            return CVerdict("generic_second_look", None, family, error="unparseable", calls=first.calls + [call])
        viol = [x for x in (d.get("violations") or []) if isinstance(x, dict)]
        acc = d["accepted"] if d["accepted"] == (not viol) else None
        return CVerdict("generic_second_look", acc, family, detail={"first": first.accepted, "violations": viol},
                        error=None if acc is not None else "inconsistent_judge_output", calls=first.calls + [call])
    return _safe("generic_second_look", family, run)


def executable(mode: str):
    """Proposed method P and its ablations as comparator callables (same signature)."""
    def fn(cand: Candidate, eo: EvidenceObject, family: Optional[str] = None, *, experiment: str = "",
           reassess_family: Optional[str] = None, dispute_family: Optional[str] = None,
           claim_level: bool = False, **_) -> CVerdict:
        import executable_verifier as EV
        v = EV.verify_executable(cand, eo, family, mode=mode, reassess_family=reassess_family, experiment=experiment,
                                 dispute_family=dispute_family, claim_level=claim_level)
        return CVerdict(f"exec_{mode}", v.accepted, family, detail=v.to_dict(), calls=v.calls,
                        error=None if v.decision != "unresolved" else v.reason)
    fn.__name__ = f"exec_{mode}"
    return fn


METHODS = {
    "always_accept": always_accept, "always_reject": always_reject, "pt_audit": pt_audit,
    "generic_judge2": generic_judge, "zandi_judge": zandi_judge, "mermigkis_verifier": mermigkis_verifier,
    "pbp_judge": pbp_judge, "generic_tool_judge": generic_tool_judge, "generic_second_look": generic_second_look,
    "exec_P": executable("P"), "exec_A1": executable("A1"), "exec_A2": executable("A2"),
    "exec_PX": executable("PX"), "exec_A0": executable("A0"),
    "exec_D": executable("D"), "exec_R": executable("R"), "exec_R11": executable("R11"),
}
from comparators2 import METHODS2 as _M2  # noqa: E402  (review baselines, aligned rubric)
METHODS.update(_M2)
