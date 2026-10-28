"""
Model registry: research identity, pinned.

A *family* is the research identity of a model (what a paper calls "the
producer" or "the judge"). A family owns exactly one canonical model and a
list of ROUTES that can serve that model. Fallback moves between routes of the
SAME family; it never moves between families.

Fidelity
--------
Routes are graded so that substitutions are never silent:

  EXACT      same published model id on a different transport
             (e.g. meta-llama/llama-3.1-8b-instruct on NanoGPT and OpenRouter)
  EQUIVALENT vendor's own aliasing of the same served model
             (e.g. deepseek-chat -> the vendor's current pointer)
  VARIANT    a different build/publisher of nominally the same model
             (e.g. unsloth/gemma-3-4b-it vs google/gemma-3-4b-it)

In strict mode the gateway refuses to cross from EXACT/EQUIVALENT to VARIANT,
because a re-quantised repack is not the same measurement instrument. The plan
this supports requires "immutable model IDs ... avoid silent model
substitution between experimental treatments".

Routing policy requested for this project
-----------------------------------------
* Llama and Gemma are served by NanoGPT first, OpenRouter second.
* OpenRouter is the universal same-model backstop for every family.
* Native vendor endpoints lead where they exist (DeepSeek, Mistral, Gemini).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from providers import PROVIDERS


class Fidelity(str, Enum):
    EXACT = "exact"
    EQUIVALENT = "equivalent"
    VARIANT = "variant"


_ORDER = {Fidelity.EXACT: 0, Fidelity.EQUIVALENT: 1, Fidelity.VARIANT: 2}


@dataclass(frozen=True)
class Route:
    """One concrete way to serve a family's model."""
    provider: str
    model_id: str                      # immutable id sent on the wire
    fidelity: Fidelity = Fidelity.EXACT
    note: str = ""

    @property
    def available(self) -> bool:
        p = PROVIDERS.get(self.provider)
        return bool(p and p.available)

    def __str__(self) -> str:
        return f"{self.provider}:{self.model_id}"


@dataclass
class Family:
    """A research identity. One model, several transports."""
    key: str                           # short id used in experiment configs
    publisher: str                     # Meta, Google, DeepSeek, ...
    display: str                       # human label for tables in the paper
    routes: List[Route] = field(default_factory=list)
    params: str = ""                   # e.g. "8B" - for the paper's model table

    def usable_routes(self, allow_variant: bool = False) -> List[Route]:
        rs = [r for r in self.routes if r.available]
        if not allow_variant:
            rs = [r for r in rs if r.fidelity is not Fidelity.VARIANT]
        return sorted(rs, key=lambda r: _ORDER[r.fidelity])

    @property
    def available(self) -> bool:
        return bool(self.usable_routes(allow_variant=True))


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
FAMILIES: Dict[str, Family] = {

    # ---- producers -------------------------------------------------------
    "deepseek": Family(
        key="deepseek", publisher="DeepSeek", display="DeepSeek Chat",
        routes=[
            Route("deepseek", "deepseek-chat", Fidelity.EQUIVALENT,
                  "native endpoint; vendor pointer to current deepseek-chat"),
            Route("nanogpt", "deepseek-chat", Fidelity.EQUIVALENT),
            Route("openrouter", "deepseek/deepseek-chat", Fidelity.EQUIVALENT),
        ],
    ),

    "mistral": Family(
        key="mistral", publisher="Mistral AI", display="Mistral Small 3.2 24B",
        params="24B",
        routes=[
            Route("mistral", "mistral-small-latest", Fidelity.EQUIVALENT,
                  "native; 'latest' is a moving pointer - pin before freezing"),
            Route("openrouter", "mistralai/mistral-small-3.2-24b-instruct", Fidelity.EXACT),
            Route("nanogpt", "mistralai/mistral-small-24b-instruct-2501", Fidelity.VARIANT,
                  "older 2501 build; not the same weights as 3.2"),
        ],
    ),

    # Llama and Gemma: NanoGPT first, OpenRouter as same-model backstop.
    "llama": Family(
        key="llama", publisher="Meta", display="Llama 3.1 8B Instruct",
        params="8B",
        routes=[
            Route("nanogpt", "meta-llama/llama-3.1-8b-instruct", Fidelity.EXACT),
            Route("openrouter", "meta-llama/llama-3.1-8b-instruct", Fidelity.EXACT),
        ],
    ),

    "llama70": Family(
        key="llama70", publisher="Meta", display="Llama 3.3 70B Instruct",
        params="70B",
        routes=[
            Route("nanogpt", "meta-llama/llama-3.3-70b-instruct", Fidelity.EXACT),
            Route("openrouter", "meta-llama/llama-3.3-70b-instruct", Fidelity.EXACT),
        ],
    ),

    # Gemma is served ONLY from Google's own build on OpenRouter. NanoGPT
    # carries unsloth repacks (unsloth/gemma-3-*-it), which are different
    # weights; they are deliberately NOT listed, so no run can reach them even
    # with allow_variant=True. One model, one identity.
    "gemma": Family(
        key="gemma", publisher="Google", display="Gemma 3 4B Instruct",
        params="4B",
        routes=[
            Route("openrouter", "google/gemma-3-4b-it", Fidelity.EXACT,
                  "Google's own build; unsloth repacks are excluded by design"),
        ],
    ),

    "gemma27": Family(
        key="gemma27", publisher="Google", display="Gemma 3 27B Instruct",
        params="27B",
        routes=[
            # Bedrock first (author's AWS credits, probed 6 Oct 2026 on both accounts)
            Route("bedrock", "google.gemma-3-27b-it", Fidelity.EXACT, "Bedrock on-demand"),
            Route("openrouter", "google/gemma-3-27b-it", Fidelity.EQUIVALENT,
                  "Google's own build; unsloth repacks are excluded by design"),
        ],
    ),

    "gemma27_vllm": Family(
        key="gemma27_vllm", publisher="Google", display="Gemma 3 27B Instruct (self-hosted vLLM)",
        params="27B",
        routes=[Route("vast_vllm", "gemma-3-27b-it", Fidelity.EXACT,
                      "google/gemma-3-27b-it bf16, vLLM on one A100 80GB (Vast.ai); token log-probabilities")]),

    # ---- judges ----------------------------------------------------------
    "gemini": Family(
        key="gemini", publisher="Google", display="Gemini 2.5 Flash Lite",
        routes=[
            Route("gemini", "gemini-2.5-flash-lite", Fidelity.EQUIVALENT,
                  "native endpoint; no same-id route elsewhere"),
        ],
    ),

    "gpt4omini": Family(
        key="gpt4omini", publisher="OpenAI", display="GPT-4o mini",
        routes=[
            Route("nanogpt", "openai/gpt-4o-mini", Fidelity.EXACT),
            Route("openrouter", "openai/gpt-4o-mini-2024-07-18", Fidelity.EXACT,
                  "dated snapshot - the immutable form of gpt-4o-mini"),
            Route("openrouter", "openai/gpt-4o-mini", Fidelity.EQUIVALENT),
        ],
    ),

    "gptoss": Family(
        key="gptoss", publisher="OpenAI", display="GPT-OSS 120B",
        params="120B",
        routes=[
            Route("nanogpt", "openai/gpt-oss-120b", Fidelity.EXACT),
            Route("openrouter", "openai/gpt-oss-120b", Fidelity.EXACT),
        ],
    ),

    # ---- capability control (frontier judge) ----------------------------
    "grok": Family(
        key="grok", publisher="xAI", display="Grok 4.3",
        routes=[
            Route("xai", "grok-4.3", Fidelity.EQUIVALENT, "native, if XAI_API_KEY set"),
            Route("nanogpt", "x-ai/grok-4.3", Fidelity.EXACT),
            Route("openrouter", "x-ai/grok-4.3", Fidelity.EXACT),
        ],
    ),

    # =====================================================================
    # STUDY ROSTER (author-selected, 2 Oct 2026). Bedrock primary, two AWS
    # accounts rotating; OpenRouter is a same-model fallback graded
    # EQUIVALENT, not EXACT: a different serving backend may differ in
    # quantisation, so a fallback-served output is logged and reportable.
    # No Anthropic models are used.
    # =====================================================================
    "nova_lite": Family(
        key="nova_lite", publisher="Amazon", display="Amazon Nova Lite",
        routes=[Route("bedrock", "amazon.nova-lite-v1:0", Fidelity.EXACT),
                Route("openrouter", "amazon/nova-lite-v1", Fidelity.EQUIVALENT)]),
    "nemotron_nano": Family(
        key="nemotron_nano", publisher="NVIDIA", display="Nemotron Nano 3 30B (A3B)",
        params="30B total / 3B active",
        routes=[Route("bedrock", "nvidia.nemotron-nano-3-30b", Fidelity.EXACT),
                Route("openrouter", "nvidia/nemotron-3-nano-30b-a3b", Fidelity.EQUIVALENT)]),
    "kimi_k25": Family(
        key="kimi_k25", publisher="Moonshot AI", display="Kimi K2.5",
        routes=[Route("bedrock", "moonshotai.kimi-k2.5", Fidelity.EXACT),
                Route("openrouter", "moonshotai/kimi-k2.5", Fidelity.EQUIVALENT)]),
    "ministral_8b": Family(
        key="ministral_8b", publisher="Mistral AI", display="Ministral 3 8B",
        params="8B",
        routes=[Route("bedrock", "mistral.ministral-3-8b-instruct", Fidelity.EXACT),
                Route("openrouter", "mistralai/ministral-8b-2512", Fidelity.EQUIVALENT)]),
    "sarvam_105b": Family(
        key="sarvam_105b", publisher="Sarvam AI", display="Sarvam-105B",
        params="105B",
        routes=[Route("sarvam", "sarvam-105b", Fidelity.EXACT,
                      "no OpenRouter fallback exists; two keys rotate. "
                      "If both fail the call raises - never substituted.")]),
    "llama4_maverick": Family(
        key="llama4_maverick", publisher="Meta", display="Llama 4 Maverick 17B",
        params="17B active",
        routes=[Route("bedrock", "us.meta.llama4-maverick-17b-instruct-v1:0", Fidelity.EXACT),
                Route("openrouter", "meta-llama/llama-4-maverick", Fidelity.EQUIVALENT)]),
    "deepseek_v32": Family(
        key="deepseek_v32", publisher="DeepSeek", display="DeepSeek V3.2",
        routes=[Route("bedrock", "deepseek.v3.2", Fidelity.EXACT),
                Route("openrouter", "deepseek/deepseek-v3.2", Fidelity.EQUIVALENT)]),
    "grok47": Family(
        key="grok47", publisher="xAI", display="Grok 4.7",
        # AWS first (author's credits), then LinkAPI, OpenRouter last (2 Oct 2026)
        routes=[Route("bedrock", "xai.grok-4.7", Fidelity.EXACT,
                      "Bedrock; agreement AVAILABLE but calls refused on both accounts on 2 Oct"),
                Route("linkapi_grok", "grok-4.7", Fidelity.EQUIVALENT,
                      "LinkAPI GrokHeavy key; response model id is grok-4.7-build"),
                Route("openrouter", "x-ai/grok-4.7", Fidelity.EQUIVALENT)]),
    # ---- strong-judge candidates (2 Oct 2026). Qualified before any use.
    # Bedrock lists Sonnet 5 and GPT-5.6 Luna on both accounts but refuses the
    # call ("not available for this account"), so OpenRouter is the only route
    # until AWS grants access; add the Bedrock route then.
    "sonnet5": Family(
        key="sonnet5", publisher="Anthropic", display="Claude Sonnet 5",
        # the author's Anthropic key first; OpenRouter as fallback (2 Oct 2026)
        routes=[Route("anthropic", "claude-sonnet-5", Fidelity.EXACT, "Anthropic API (workspace-scoped)"),
                Route("openrouter", "anthropic/claude-sonnet-5", Fidelity.EQUIVALENT, "fallback")]),
    "gpt56_luna": Family(
        key="gpt56_luna", publisher="OpenAI", display="GPT-5.6 Luna",
        routes=[Route("openai_direct", "gpt-5.6-luna", Fidelity.EXACT, "OpenAI API"),
                Route("openrouter", "openai/gpt-5.6-luna", Fidelity.EQUIVALENT, "fallback")]),
    "grok45": Family(
        key="grok45", publisher="xAI", display="Grok 4.5 (reasoning, via LinkAPI)",
        # 7 Oct 2026: the author asked for Grok through LinkAPI only (no xAI API). LinkAPI does not
        # serve grok-4.20 on these keys; grok-4.5 is the cheapest Grok it serves (ratio 1, completion 3;
        # GrokHeavy group 0.25 -> $0.50/$1.50 per M; GrokBuild group 0.05 -> $0.10/$0.30 per M).
        routes=[Route("linkapi_grok", "grok-4.5", Fidelity.EQUIVALENT, "LinkAPI GrokHeavy group"),
                Route("linkapi_grokbuild", "grok-4.5", Fidelity.EQUIVALENT, "LinkAPI GrokBuild group (slow)")]),
    "grok420": Family(
        key="grok420", publisher="xAI", display="Grok 4.20 (non-reasoning)",
        # OpenRouter's x-ai/grok-4.20 is the non-reasoning build: identical token
        # usage and cost to xAI's grok-4.20-0309-non-reasoning on the same prompt
        routes=[Route("xai_direct", "grok-4.20-0309-non-reasoning", Fidelity.EXACT, "xAI API"),
                Route("openrouter", "x-ai/grok-4.20", Fidelity.EQUIVALENT,
                      "fallback; verified to serve the non-reasoning build")]),
    "gemini31_pro": Family(
        key="gemini31_pro", publisher="Google", display="Gemini 3.1 Pro (preview)",
        routes=[Route("linkapi_gemini", "gemini-3.1-pro-preview", Fidelity.EQUIVALENT,
                      "LinkAPI aggregator"),
                Route("openrouter", "google/gemini-3.1-pro-preview", Fidelity.EQUIVALENT)]),
    # ---- budget tier T1: cheap in-cloud judges on Bedrock (probed 6 Oct 2026; both
    # accounts serve them). Publishers differ from all four producers, so every
    # producer-judge pair stays cross-publisher.
    "qwen3_next": Family(
        key="qwen3_next", publisher="Alibaba (Qwen)", display="Qwen3-Next 80B-A3B",
        params="80B total / 3B active",
        routes=[Route("bedrock", "qwen.qwen3-next-80b-a3b", Fidelity.EXACT, "Bedrock on-demand"),
                Route("openrouter", "qwen/qwen3-next-80b-a3b-instruct", Fidelity.EQUIVALENT, "fallback")]),
    "glm47": Family(
        key="glm47", publisher="Z.AI", display="GLM-4.7",
        routes=[Route("bedrock", "zai.glm-4.7", Fidelity.EXACT, "Bedrock on-demand"),
                Route("openrouter", "z-ai/glm-4.7", Fidelity.EQUIVALENT, "fallback")]),
    "gptoss120b": Family(
        key="gptoss120b", publisher="OpenAI (open weights)", display="GPT-OSS 120B",
        params="117B total / 5.1B active",
        routes=[Route("bedrock", "openai.gpt-oss-120b-1:0", Fidelity.EXACT,
                      "Bedrock on-demand; a reasoning model: give it a large max_tokens"),
                Route("openrouter", "openai/gpt-oss-120b", Fidelity.EQUIVALENT, "fallback")]),
    "gpt5_mini": Family(
        key="gpt5_mini", publisher="OpenAI", display="GPT-5 mini",
        routes=[Route("linkapi", "gpt-5-mini", Fidelity.EQUIVALENT,
                      "LinkAPI aggregator (Azure OpenAI); backend version not exposed")]),
}


# ---------------------------------------------------------------------------
# Experiment role assignment (edit here, not in experiment scripts)
# ---------------------------------------------------------------------------
#: v1 pilot (already run, 24 Taiwan borrowers) - kept for provenance only.
PILOT_PRODUCERS: List[str] = ["deepseek", "llama"]
PILOT_JUDGES: List[str] = ["gemini", "gpt4omini"]

#: STUDY ROSTER, author-selected 2 Oct 2026; revised same day.
#: DeepSeek V3.2 and Sarvam-105B were DROPPED by the author as problematic
#: (DeepSeek V3.2 rejected a trivially correct case in the smoke test; Sarvam-105B
#: spent ~400 reasoning tokens on a one-line check and has no fallback route).
#: Their families stay registered for provenance but are not used.
#: Four audited producers, four publishers, spanning a capability range:
#:   small:  nova_lite (Amazon), ministral_8b (Mistral), nemotron_nano (NVIDIA, 3B active)
#:   large:  kimi_k25 (Moonshot)
AUDITED_PRODUCERS: List[str] = ["nova_lite", "nemotron_nano", "kimi_k25",
                                "ministral_8b"]
#: JUDGES - chosen 2 Oct 2026 by the pre-registered rule in qualify_judges.py
#: (eligible: unresolved <= 5%, false-reject b <= 0.20 on 120 adjudicated
#: natural items; primary = lowest false-accept a; second = best eligible from
#: another publisher). Result in Submission/judge_qualification.json:
#:   grok47   a=0.24 b=0.053   sonnet5 a=0.28 b=0.032   gemini31_pro a=0.28 b=0.053
#:   gpt5_mini a=0.32 b=0.053 gpt56_luna a=0.32 b=0.074 llama4_maverick a=0.00 b=0.59 (INELIGIBLE)
#: The a values differ by one item on 25 errors (95% CIs overlap fully), and
#: with the structural checker in front every eligible judge accepts 0-1/25.
#: Llama 4 Maverick and GPT-5-mini (the earlier roster) are kept registered.
# Re-qualified 2 Oct 2026 on the direct platforms (judge_qualification_prompt2.json):
#   gpt56_luna a=0.16 b=0.074 | grok420 a=0.28 b=0.084 | sonnet5 a=0.28 b=0.053 (all eligible)
# The pre-registered rule selects Luna (lowest a) as primary, Grok 4.20 as second.
# All three judge every M1 item; the roles matter for the release policy only.
PRIMARY_JUDGE: str = "gpt56_luna"
#: Second judge on a prespecified 20% of pairs, and judge-output parse repair.
SECOND_JUDGE: str = "grok420"
SECOND_JUDGE_FRACTION: float = 0.20
#: Strong-judge condition: Sonnet 5 is the most expensive judge (about $13.5 per
#: 1,000 M1 calls against about $1 for Luna), reported alongside so judge
#: strength is a factor, and the strong stage of the cheap-then-strong cascade
#: (comparators.py). The cheapest judge, Luna, is primary by the qualification rule.
CANDIDATE_JUDGE: str = "sonnet5"
#: The cascade's stages (Kiyani et al.-style comparator; budget-matched).
CASCADE_WEAK_JUDGE: str = "gpt56_luna"
CASCADE_STRONG_JUDGE: str = "sonnet5"
#: Parser judge: converts an unparseable producer/judge response into the
#: required JSON. It only copies; json_parse.faithful() rejects any output that
#: adds, rewords or infers content, and the item is then UNRESOLVED.
#: 6 Oct 2026: LinkAPI's Azure OpenAI balance is exhausted, so the parser moved to
#: Gemma 3 27B on Bedrock (Google: not a producer or judge publisher in the study).
#: Responses parsed by GPT-5 mini before this date keep their recorded parser id.
PARSER_FAMILY: str = "gemma27"
#: Qualified pool (all six were run; see the JSON above).
JUDGE_CANDIDATES: List[str] = ["sonnet5", "gpt56_luna", "gemini31_pro", "grok47", "grok420"]

#: Budget tiers for the verifier factorial (6 Oct 2026). Every verification METHOD
#: is run with the same judge model per tier, so method and model effects separate.
#:   T0  deterministic (structural + contract detectors)   $0
#:   T1  cheap in-cloud judge on Bedrock                     ~$1 per 1,000 calls
#:   T2  low-cost external judge (xAI, non-reasoning)        ~$4 per 1,000 calls (M1 actual)
#:   T3  premium judges, already run once on all M1 items    ~$1-14 per 1,000 calls
#:   S   Indian-hosted (data-residency) judge, small subset  reasoning model, subset only
T1_JUDGE_CANDIDATES: List[str] = ["qwen3_next", "glm47", "gptoss120b"]
T2_JUDGE: str = "grok45"   # 7 Oct 2026: Grok via LinkAPI only; grok420 (xAI) is historical (M1 judging)
T3_JUDGES: List[str] = ["gpt56_luna", "sonnet5"]
SOVEREIGN_JUDGE: str = "sarvam_105b"

MAIN_PRODUCERS: List[str] = AUDITED_PRODUCERS
MAIN_JUDGES: List[str] = [j for j in (PRIMARY_JUDGE, SECOND_JUDGE, CANDIDATE_JUDGE) if j]
CONTROL_JUDGES: List[str] = []
ALT_GOOGLE_JUDGE: str = "gemma"
SAME_PUBLISHER_CONTROL: List[tuple] = []


def get(key: str) -> Family:
    if key not in FAMILIES:
        raise KeyError(f"unknown family {key!r}; known: {sorted(FAMILIES)}")
    return FAMILIES[key]


def distinct_publishers(keys: List[str]) -> bool:
    """True when no two families share a publisher (needed for diversity claims)."""
    pubs = [get(k).publisher for k in keys]
    return len(set(pubs)) == len(pubs)


def cross_family_pairs(producers: List[str], judges: List[str]) -> List[tuple]:
    """(producer, judge) pairs where the judge is a different PUBLISHER.

    Publisher, not family: Gemini and Gemma are both Google, so a Gemma judge
    on a Gemini-produced explanation is not an independent check.
    """
    out = []
    for p in producers:
        for j in judges:
            if get(p).publisher != get(j).publisher:
                out.append((p, j))
    return out


class PanelNotIndependent(RuntimeError):
    """The judge panel cannot independently audit these producers."""


def assert_independent_panel(producers: List[str], judges: List[str],
                             *, require_full_matrix: bool = True) -> dict:
    """Hard gate, called at experiment start - not an advisory check.

    Three conditions, each of which has silently broken a real design:
      1. no judge shares a publisher with ANY producer;
      2. judges are mutually distinct publishers (otherwise 'three judges' is
         really one opinion counted twice);
      3. with require_full_matrix, every producer x judge cell is usable.
    """
    problems: List[str] = []
    prod_pubs = {p: get(p).publisher for p in producers}
    judge_pubs = {j: get(j).publisher for j in judges}

    overlap = {j: pub for j, pub in judge_pubs.items() if pub in prod_pubs.values()}
    for j, pub in overlap.items():
        shared = [p for p, pp in prod_pubs.items() if pp == pub]
        problems.append(
            f"judge {j!r} shares publisher {pub!r} with producer(s) {shared}")

    if len(set(judge_pubs.values())) != len(judges):
        dupes = [pub for pub in set(judge_pubs.values())
                 if list(judge_pubs.values()).count(pub) > 1]
        problems.append(f"judges are not mutually independent; duplicated "
                        f"publisher(s) {dupes} in {judge_pubs}")

    unusable = [f for f in producers + judges if not get(f).available]
    if unusable:
        problems.append(f"no reachable route for {unusable}")

    pairs = cross_family_pairs(producers, judges)
    expected = len(producers) * len(judges)
    if require_full_matrix and len(pairs) != expected:
        problems.append(f"judge matrix incomplete: {len(pairs)}/{expected} cells")

    if problems:
        raise PanelNotIndependent(
            "judge panel is not independent:\n  - " + "\n  - ".join(problems))
    return {"producers": prod_pubs, "judges": judge_pubs,
            "cross_publisher_pairs": len(pairs), "full_matrix": len(pairs) == expected}


def manifest(keys: Optional[List[str]] = None) -> List[dict]:
    """Machine-readable record of exactly what was used - goes in the paper."""
    keys = keys or sorted(FAMILIES)
    out = []
    for k in keys:
        f = get(k)
        out.append({
            "family": f.key,
            "publisher": f.publisher,
            "display": f.display,
            "params": f.params,
            "available": f.available,
            "routes": [{"provider": r.provider, "model_id": r.model_id,
                        "fidelity": r.fidelity.value, "available": r.available,
                        "note": r.note} for r in f.routes],
        })
    return out


if __name__ == "__main__":
    import json
    print("FAMILY        PUBLISHER   AVAIL  ROUTES (in fallback order)")
    print("-" * 100)
    for k in sorted(FAMILIES):
        f = FAMILIES[k]
        rs = " -> ".join(
            f"{r}[{r.fidelity.value[:3]}{'' if r.available else ',OFF'}]" for r in f.routes)
        print(f"  {f.key:<12}{f.publisher:<11}{str(f.available):<7}{rs}")
    print()
    print("pilot producers :", PILOT_PRODUCERS, "distinct publishers:", distinct_publishers(PILOT_PRODUCERS))
    print("pilot judges    :", PILOT_JUDGES, "distinct publishers:", distinct_publishers(PILOT_JUDGES))
    print("main producers  :", MAIN_PRODUCERS, "distinct publishers:", distinct_publishers(MAIN_PRODUCERS))
    print("main judges     :", MAIN_JUDGES, "distinct publishers:", distinct_publishers(MAIN_JUDGES))
    print("cross-publisher pairs (main):", len(cross_family_pairs(MAIN_PRODUCERS, MAIN_JUDGES)))
