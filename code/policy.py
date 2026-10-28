"""
Group-audited release policy, metrics and certificate (plan §4.3-4.5).

This is where the paper's research question is actually answered: when a
verifier filters explanations, does it reduce error for every group, or buy a
better average by withholding useful explanations from some groups?

Five reported quantities per group g and policy lambda (all five, always):

  C_g  narrative coverage          P(A=1 | g)
  R_g  selective error             P(L=1 | A=1, g)        undefined if C_g = 0
  H_g  unsafe narrative exposure   P(A=1, L=1 | g) = C_g R_g
  U_g  useful final-explanation    correct AND informative, narrative OR template
  B_g  burden                      template use, regenerations, escalations, latency

Two traps this module refuses to fall into:
  * selective error improves by excluding hard cases -> C_g is always shown;
  * exposure improves by releasing nothing -> R_g with C_g = 0 is None, never 0.

The loss L must come from independent labels, never from the production judge.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# One evaluated record
# ---------------------------------------------------------------------------
@dataclass
class Record:
    borrower_id: str
    group: str                      # prespecified audit group
    arm: str
    family: str
    released: bool                  # A: narrative released (set by a policy)
    loss: Optional[bool]            # L: independently labelled material violation
    informative: Optional[bool] = None   # content floor met
    template_used: bool = False
    regenerations: int = 0
    escalated: bool = False
    latency_ms: int = 0
    cost_usd: float = 0.0
    # runtime signals a policy may use (group attributes are NOT among them)
    completeness: float = 1.0
    structural_ok: bool = True
    judge_accept: Optional[bool] = None
    judge_violations: int = 0
    # template quality, for the useful-rate denominator
    template_correct: Optional[bool] = None
    template_informative: Optional[bool] = None


# ---------------------------------------------------------------------------
# Policies: frozen family, defined on development data only
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Policy:
    name: str
    min_completeness: float = 0.0
    require_structural: bool = True
    require_judge: bool = False
    max_judge_violations: int = 0

    def release(self, r: Record) -> bool:
        if r.completeness < self.min_completeness:
            return False
        if self.require_structural and not r.structural_ok:
            return False
        if self.require_judge:
            if r.judge_accept is not True:
                return False
            if r.judge_violations > self.max_judge_violations:
                return False
        return True


def policy_family(completeness_grid: Sequence[float] = (0.0, 0.5, 0.7, 0.9)
                  ) -> List[Policy]:
    """A small, frozen set. Must be fixed before looking at test data."""
    out = [Policy("release_all", 0.0, False, False)]
    for c in completeness_grid:
        out.append(Policy(f"struct@c{c}", c, True, False))
        out.append(Policy(f"struct+judge@c{c}", c, True, True))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
@dataclass
class GroupMetrics:
    group: str
    n: int
    released: int
    coverage: float                       # C_g
    selective_error: Optional[float]      # R_g - None when nothing released
    exposure: float                       # H_g
    useful_rate: float                    # U_g
    template_rate: float
    escalation_rate: float
    mean_regenerations: float
    mean_latency_ms: float
    total_cost_usd: float
    labelled: int                         # how many have an independent label
    # exact integer counts (certify() must never reconstruct them from rounded rates)
    k_errors: int = 0                     # released AND labelled erroneous
    released_unlabelled: int = 0          # released, label unknown
    useful: int = 0                       # known correct AND known informative
    useful_unknown: int = 0               # usefulness not established (unknown label/content)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def evaluate(records: List[Record], policy: Policy,
             *, content_floor: bool = True) -> Dict[str, GroupMetrics]:
    """Apply a policy and compute the five quantities per group."""
    by_g: Dict[str, List[Record]] = {}
    for r in records:
        by_g.setdefault(r.group, []).append(r)

    out: Dict[str, GroupMetrics] = {}
    for g, rs in sorted(by_g.items()):
        n = len(rs)
        rel = [r for r in rs if policy.release(r)]
        c = len(rel) / n if n else 0.0

        lab = [r for r in rel if r.loss is not None]
        r_g = (sum(1 for r in lab if r.loss) / len(lab)) if lab else None
        h_g = (c * r_g) if r_g is not None else 0.0

        # useful = KNOWN correct AND KNOWN informative, narrative or template.
        # An unknown label or unknown content is not counted as useful; it is
        # reported separately (useful_unknown) so a sensitivity bound is possible.
        useful = unknown = 0
        for r in rs:
            if policy.release(r):
                ok, inf = r.loss, (r.informative if content_floor else True)
                ok = None if ok is None else (not ok)
            else:
                ok = r.template_correct
                inf = r.template_informative if content_floor else True
            if ok is True and inf is True:
                useful += 1
            elif ok is not False and inf is not False:
                unknown += 1
        u_g = useful / n if n else 0.0

        out[g] = GroupMetrics(
            group=g, n=n, released=len(rel), coverage=round(c, 4),
            selective_error=(round(r_g, 4) if r_g is not None else None),
            exposure=round(h_g, 4), useful_rate=round(u_g, 4),
            template_rate=round(sum(1 for r in rs if not policy.release(r)) / n, 4) if n else 0.0,
            escalation_rate=round(sum(1 for r in rs if r.escalated) / n, 4) if n else 0.0,
            mean_regenerations=round(sum(r.regenerations for r in rs) / n, 3) if n else 0.0,
            mean_latency_ms=round(sum(r.latency_ms for r in rs) / n, 1) if n else 0.0,
            total_cost_usd=round(sum(r.cost_usd for r in rs), 4),
            labelled=len(lab), k_errors=sum(1 for r in lab if r.loss),
            released_unlabelled=len(rel) - len(lab), useful=useful, useful_unknown=unknown)
    return out


def summarise(m: Dict[str, GroupMetrics]) -> dict:
    """Worst-group view. Groups with nothing released cannot report an error."""
    cov = [x.coverage for x in m.values()]
    use = [x.useful_rate for x in m.values()]
    errs = [x.selective_error for x in m.values() if x.selective_error is not None]
    undefined = [g for g, x in m.items() if x.selective_error is None]
    return {
        "min_coverage": round(min(cov), 4) if cov else 0.0,
        "coverage_gap": round(max(cov) - min(cov), 4) if cov else 0.0,
        "worst_group_error": round(max(errs), 4) if errs else None,
        "max_exposure": round(max(x.exposure for x in m.values()), 4) if m else 0.0,
        "min_useful_rate": round(min(use), 4) if use else 0.0,
        "useful_rate_gap": round(max(use) - min(use), 4) if use else 0.0,
        "groups_with_undefined_error": undefined,
    }


# ---------------------------------------------------------------------------
# Selection: max-min coverage among policies meeting a worst-group error bound
# ---------------------------------------------------------------------------
def select_policy(records: List[Record], policies: List[Policy],
                  *, eps: float, min_coverage: float = 0.0,
                  max_coverage_gap: float = 1.0,
                  use_certificate: bool = False, delta: float = 0.05
                  ) -> Tuple[Optional[Policy], List[dict]]:
    """Choose on DEVELOPMENT data only. Returns (policy or None, audit rows).

    Infeasibility is a reportable outcome, not a reason to relax eps after
    seeing results.
    """
    groups = sorted({r.group for r in records})
    rows: List[dict] = []
    best: Optional[Tuple[float, Policy]] = None

    for p in policies:
        m = evaluate(records, p)
        s = summarise(m)
        if use_certificate:
            cert = certify(m, policies_m=len(policies), groups_g=len(groups),
                           delta=delta, eps=eps)
            ok_err = cert["all_groups_below_eps"]
            bound = cert["max_upper_bound"]
        else:
            wge = s["worst_group_error"]
            ok_err = (wge is not None and wge <= eps)
            bound = wge
        feasible = (ok_err and s["min_coverage"] >= min_coverage
                    and s["coverage_gap"] <= max_coverage_gap
                    and not s["groups_with_undefined_error"])
        rows.append({"policy": p.name, "feasible": feasible,
                     "bound": bound, **s})
        if feasible and (best is None or s["min_coverage"] > best[0]):
            best = (s["min_coverage"], p)
    return (best[1] if best else None), rows


# ---------------------------------------------------------------------------
# Clopper-Pearson certificate (plan §4.4)
# ---------------------------------------------------------------------------
def clopper_pearson_upper(k: int, n: int, alpha: float) -> float:
    """One-sided upper bound on a binomial rate. Returns 1.0 when n = 0."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    try:
        from scipy.stats import beta
        return float(beta.ppf(1 - alpha, k + 1, n - k))
    except Exception:
        if k == 0:                       # exact closed form
            return 1.0 - alpha ** (1.0 / n)
        lo, hi = k / n, 1.0
        for _ in range(200):             # bisection on the binomial tail
            mid = (lo + hi) / 2
            tail = sum(math.comb(n, i) * mid ** i * (1 - mid) ** (n - i)
                       for i in range(0, k + 1))
            if tail > alpha:
                lo = mid
            else:
                hi = mid
        return hi


def certify(metrics: Dict[str, GroupMetrics], *, policies_m: int, groups_g: int,
            delta: float = 0.05, eps: float = 0.05) -> dict:
    """Simultaneous per-group upper bounds with delta/(M*G) allocation.

    Scope, stated in the paper verbatim: this concerns the defined binary
    narrative loss, for prespecified groups, under independent sampling from
    the same distribution, with a frozen generator/checker/label protocol. It
    is NOT a fairness certificate, not an individual guarantee, not valid for
    arbitrary subgroups, and not valid after a model or distribution change.
    """
    alpha = delta / max(1, policies_m * groups_g)
    per: Dict[str, dict] = {}
    for g, m in metrics.items():
        # exact counts; a released case with no label is counted as an error
        # (conservative), never dropped from the denominator
        n = m.labelled + m.released_unlabelled
        k = m.k_errors + m.released_unlabelled
        ub = clopper_pearson_upper(k, n, alpha)
        per[g] = {"n_released": n, "k_violations": k, "k_labelled_errors": m.k_errors,
                  "released_unlabelled_counted_as_errors": m.released_unlabelled,
                  "upper_bound": round(ub, 5), "below_eps": bool(ub <= eps)}
    ubs = [v["upper_bound"] for v in per.values()] or [1.0]
    return {"alpha_per_test": alpha, "delta": delta, "eps": eps,
            "policies_m": policies_m, "groups_g": groups_g,
            "per_group": per, "max_upper_bound": round(max(ubs), 5),
            "all_groups_below_eps": all(v["below_eps"] for v in per.values())}


def required_n(eps: float, alpha: float) -> int:
    """Released labelled cases per group needed for a zero-error certificate."""
    n = 1
    while n < 10 ** 7:
        if 1.0 - alpha ** (1.0 / n) <= eps:
            return n
        n += 1
    return -1


# ---------------------------------------------------------------------------
# Mechanistic decomposition (plan §4.5)
# ---------------------------------------------------------------------------
def decompose(records: List[Record]) -> Dict[str, dict]:
    """Per group: p_g (candidate error rate), a_g (accept an erroneous one),
    b_g (reject a correct one), and the implied C_g / R_g.

        C_g = p_g a_g + (1 - p_g)(1 - b_g)
        R_g = p_g a_g / C_g

    An accounting identity, not a theorem. It shows whether a coverage gap is
    driven by generation quality or by verifier behaviour.
    """
    out: Dict[str, dict] = {}
    by_g: Dict[str, List[Record]] = {}
    for r in records:
        by_g.setdefault(r.group, []).append(r)

    for g, rs in sorted(by_g.items()):
        lab = [r for r in rs if r.loss is not None and r.judge_accept is not None]
        if not lab:
            out[g] = {"n_labelled": 0}
            continue
        err = [r for r in lab if r.loss]
        cor = [r for r in lab if not r.loss]
        p = len(err) / len(lab)
        a = (sum(1 for r in err if r.judge_accept) / len(err)) if err else None
        b = (sum(1 for r in cor if not r.judge_accept) / len(cor)) if cor else None
        c = (p * (a or 0.0)) + ((1 - p) * (1 - (b or 0.0)))
        out[g] = {"n_labelled": len(lab),
                  "p_candidate_error": round(p, 4),
                  "a_false_accept": (round(a, 4) if a is not None else None),
                  "b_false_reject": (round(b, 4) if b is not None else None),
                  "implied_coverage": round(c, 4),
                  "implied_selective_error":
                      (round(p * (a or 0.0) / c, 4) if c > 0 else None)}
    return out


# ---------------------------------------------------------------------------
def report(records: List[Record], policy: Policy, *, policies_m: int = 1,
           delta: float = 0.05, eps: float = 0.05) -> dict:
    m = evaluate(records, policy)
    return {"policy": policy.name,
            "per_group": {g: v.to_dict() for g, v in m.items()},
            "summary": summarise(m),
            "certificate": certify(m, policies_m=policies_m,
                                   groups_g=len(m), delta=delta, eps=eps),
            "decomposition": decompose(records)}


if __name__ == "__main__":
    import random
    rng = random.Random(7)
    recs: List[Record] = []
    # group B deliberately gets a harsher judge: same generator quality,
    # more false rejections -> the exact effect the paper is looking for
    for g, p_err, b_fr in (("A", 0.20, 0.05), ("B", 0.20, 0.35)):
        for i in range(150):
            loss = rng.random() < p_err
            judge_ok = (rng.random() < 0.30) if loss else (rng.random() > b_fr)
            recs.append(Record(
                borrower_id=f"{g}-{i}", group=g, arm="typed", family="llama",
                released=True, loss=loss, informative=True,
                completeness=rng.choice([0.6, 0.8, 1.0]),
                structural_ok=rng.random() > 0.05,
                judge_accept=judge_ok, judge_violations=0 if judge_ok else 1,
                template_correct=True, template_informative=True,
                latency_ms=rng.randint(800, 3000), cost_usd=0.0009))

    pols = policy_family()
    chosen, rows = select_policy(recs, pols, eps=0.10, min_coverage=0.50)
    print(f"{'policy':<22}{'feas':<6}{'minCov':<8}{'covGap':<8}{'worstErr':<10}{'minUseful'}")
    for r in rows:
        print(f"  {r['policy']:<20}{str(r['feasible']):<6}{r['min_coverage']:<8}"
              f"{r['coverage_gap']:<8}{str(r['worst_group_error']):<10}{r['min_useful_rate']}")
    print(f"\nselected: {chosen.name if chosen else 'INFEASIBLE'}")
    if chosen:
        rep = report(recs, chosen, policies_m=len(pols))
        print(json.dumps(rep["summary"], indent=2))
        print("decomposition:", json.dumps(rep["decomposition"], indent=2))
    print(f"\nn per group for zero-error cert at eps=5%, alpha=0.0125: "
          f"{required_n(0.05, 0.0125)}")
