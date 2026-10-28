"""
JSON contract for every LLM response, with a parser judge as last resort.

Every producer and judge must answer in JSON. Parsing runs in three steps:

  1. direct     reasoning blocks stripped; the whole text, else EVERY balanced
                top-level {...} is tried. Exactly one schema-valid object (or
                several identical ones) is required; an echoed schema
                placeholder is ignored; two different valid objects (a draft
                and a final answer) make the response UNRESOLVED, never a guess.
  2. repair     json_repair, only when there is exactly one candidate object,
                no schema placeholder in it, and the call finished cleanly.
  3. parser     a separate LLM (registry.PARSER_FAMILY, not a study judge or
                producer) is shown the response and the schema and asked to COPY
                it into JSON. Accepted only if, deterministically:
                  - faithful: every string it returns occurs in the response
                    (JSON escapes decoded, quotes/hyphens/case normalised); a
                    judge verdict must be stated literally in the response;
                    typed claims keep the evidence ids written next to them;
                  - complete: >= 90% (multiset) of the response's content tokens,
                    including numbers and evidence ids, are kept; for a judge,
                    an accepting verdict carries no violations and a rejecting
                    one keeps every violation type the response names;
                  - the parser call itself finished cleanly and parsed directly.

Finish reasons are normalised in one place (classify_finish). A truncated
response is an OBSERVATION (method 'truncated'), never repaired; a filtered
one is 'filtered'; a parser outage is 'parser_unavailable' (a transport
failure the run may retry), distinct from an unfaithful/invalid parse.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

PARSER_SYS = (
    "You convert another model's response into JSON that matches a given schema. "
    "Copy text exactly as it appears in the response. Never add, remove, reword or "
    "summarise content; never change numbers; never infer a value that the response "
    "does not state explicitly. If a required value is not stated, use null. "
    "Reply with the JSON object only.")

_TRUNC = {"length", "max_tokens", "max_output_tokens", "model_context_window_exceeded",
          "token_limit"}
_FILTER = {"content_filter", "content_filtered", "guardrail_intervened", "safety",
           "recitation", "blocked"}
_CLEAN = {"stop", "end_turn", "stop_sequence", "eos", "tool_calls", "function_call"}


def classify_finish(fr: Optional[str]) -> str:
    """'clean' | 'truncated' | 'filtered' | 'unknown' for any provider's value."""
    f = (fr or "").strip().lower()
    if f in _TRUNC:
        return "truncated"
    if f in _FILTER:
        return "filtered"
    if f in _CLEAN:
        return "clean"
    return "unknown"


# a schema ECHO carries the schema's literal placeholders; ordinary '...' or '<10%'
# in a real answer must not make it look like one
_PLACEHOLDER_STR = re.compile(r'"<(?:type|text|id|exact words|one sentence[^"]*|the [^"]*verbatim|'
                              r'concept or null|one allowed concept or null)>"')


def strip_reasoning(raw: str) -> str:
    """Remove reasoning: <think>...</think> blocks, and everything up to the LAST
    closing tag when the opening tag was dropped by the route."""
    t = raw or ""
    t = re.sub(r"<(think|thinking|reasoning)>.*?</\1>", "", t, flags=re.S | re.I)
    m = list(re.finditer(r"</(think|thinking|reasoning)>", t, flags=re.I))
    if m:
        t = t[m[-1].end():]
    return t.strip()


def _strip_fences(t: str) -> str:
    t = re.sub(r"```(?:json)?\s*", "", t.strip(), flags=re.I)
    return t.replace("```", "")


def all_objects(text: str) -> List[str]:
    """Every balanced top-level {...} span, in order."""
    out, i = [], 0
    while True:
        s = text.find("{", i)
        if s == -1:
            return out
        depth, instr, esc, end = 0, False, False, None
        for k in range(s, len(text)):
            ch = text[k]
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
                    end = k
                    break
        if end is None:
            out.append(text[s:])          # unbalanced tail (truncated)
            return out
        out.append(text[s:end + 1])
        i = end + 1


def first_object(text: str) -> Optional[str]:
    objs = all_objects(text)
    return objs[0] if objs else None


def looks_truncated(raw: str) -> bool:
    """The last JSON object never closes: the text was cut off. Uses the same
    scanner as all_objects (escape-aware), which appends an unclosed tail."""
    t = _strip_fences(strip_reasoning(raw or ""))
    objs = all_objects(t)
    return bool(objs) and t.rstrip().endswith(objs[-1].rstrip()) and not _closes(objs[-1])


def _closes(span: str) -> bool:
    depth, instr, esc = 0, False, False
    for ch in span:
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
    return depth == 0 and not instr


class _DupKey(ValueError):
    pass


def _no_conflicting_dups(pairs):
    """object_pairs_hook: a key stated twice with different values ('accepted':
    false ... 'accepted': true) is a self-contradiction, never last-wins."""
    out = {}
    for k, v in pairs:
        if k in out and json.dumps(out[k], sort_keys=True) != json.dumps(v, sort_keys=True):
            raise _DupKey(k)
        out[k] = v
    return out


def _loads(t: str):
    return json.loads(t, object_pairs_hook=_no_conflicting_dups)


def _is_echo(span: str) -> bool:
    """A copy of the SCHEMA itself: placeholders outside string values, or a
    string value that is exactly a schema placeholder. '[...]' or '...' inside a
    real quote does not make an answer an echo."""
    blanked = re.sub(r'"(?:[^"\\]|\\.)*"', '""', span)
    return bool(re.search(r"true\|false|\[\s*\.\.\.\s*\]", blanked) or _PLACEHOLDER_STR.search(span))


def parse_local(raw: str, validate: Optional[Callable[[Any], bool]] = None,
                allow_repair: bool = True) -> Tuple[Optional[Any], str]:
    """Steps 1-2. Returns (object or None, method). method in
    direct | repair | ambiguous | empty | unparsed."""
    if not raw or not raw.strip():
        return None, "empty"
    ok = validate or (lambda o: isinstance(o, (dict, list)) and bool(o))
    t = _strip_fences(strip_reasoning(raw))
    try:
        o = _loads(t)
        if ok(o):
            return o, "direct"
    except _DupKey:
        return None, "ambiguous"
    except Exception:  # noqa: BLE001
        pass
    objs = [x for x in all_objects(t) if not _is_echo(x)]
    valid = []
    for x in objs:
        try:
            o = _loads(x)
        except _DupKey:
            return None, "ambiguous"
        except Exception:  # noqa: BLE001
            continue
        if ok(o):
            valid.append(o)
    distinct = {json.dumps(o, sort_keys=True) for o in valid}
    if len(distinct) == 1:
        return valid[0], "direct"
    if len(distinct) > 1:
        return None, "ambiguous"        # a draft and a final answer: never guess
    if allow_repair and len(objs) == 1:
        try:
            from json_repair import repair_json
            o = _loads(repair_json(objs[0]))
            if ok(o):
                return o, "repair"
        except Exception:  # noqa: BLE001
            pass
    return None, "unparsed"


# ---------------------------------------------------------------------------
# faithfulness and completeness of a parser output
# ---------------------------------------------------------------------------
_TRANS = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                        "‐": "-", "‑": "-", "‒": "-", "–": "-",
                        "—": "-", " ": " ", " ": " "})


def _decoded(raw: str) -> str:
    """raw plus every JSON string literal in it decoded (\\u2019, \\n, \\" ...)."""
    extra = []
    for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', raw):
        try:
            extra.append(json.loads(f'"{m.group(1)}"'))
        except Exception:  # noqa: BLE001
            pass
    return raw + "\n" + "\n".join(extra)


def _decode_inplace(raw: str) -> str:
    """raw with every JSON string literal decoded where it stands (structure kept)."""
    def dec(m):
        try:
            return '"' + json.loads(m.group(0)) + '"'
        except Exception:  # noqa: BLE001
            return m.group(0)
    return re.sub(r'"(?:[^"\\]|\\.)*"', dec, raw)


def _norm(s: str) -> str:
    s = (s or "").translate(_TRANS).replace('\\"', '"').replace("\\n", " ")
    s = re.sub(r"[*`]{1,3}", "", s)             # markdown emphasis
    s = s.replace("_", " ")                     # 'sign_reversal' == 'sign reversal' 
    return re.sub(r"\s+", " ", s).strip().lower()


def _norm_lines(s: str) -> str:
    """_norm, but line breaks survive (for line/list-item units in prose)."""
    s = (s or "").translate(_TRANS).replace('\\"', '"').replace("\\n", "\n")
    s = re.sub(r"[*`]{1,3}", "", s).replace("_", " ")
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return re.sub(r" ?\n ?", "\n", s).strip().lower()


def _strings(o) -> List[str]:
    if isinstance(o, str):
        return [o]
    if isinstance(o, dict):
        return [s for v in o.values() for s in _strings(v)]
    if isinstance(o, list):
        return [s for v in o for s in _strings(v)]
    return []


def _verdict_literal(raw: str, key: str) -> Optional[str]:
    raw = re.sub(r"[*`_]{1,3}(?=\s*[:=])|(?<=[:=])\s*[*`_]{1,3}", "", raw)   # '**Accepted**: false'
    m = re.findall(rf"{key}['\"]?\s*[:=]\s*['\"]?(true|false)\b", raw, flags=re.I)
    vals = {x.lower() for x in m}
    return vals.pop() if len(vals) == 1 else None


def faithful(obj: Any, raw: str, verdict_key: Optional[str] = None) -> Tuple[bool, str]:
    """The parser output only COPIES the response."""
    src = strip_reasoning(raw)
    nr = _norm(_decoded(src))
    for s in _strings(obj):
        ns = _norm(s)
        if ns and ns not in nr:
            return False, f"string not in response: {s[:60]!r}"
    if verdict_key and isinstance(obj, dict) and verdict_key in obj:
        lit = _verdict_literal(src, verdict_key)
        if lit is None:
            return False, f"'{verdict_key}' not stated explicitly (or stated both ways) in response"
        if str(obj[verdict_key]).lower() != lit:
            return False, f"'{verdict_key}' differs from response"
    if isinstance(obj, dict) and isinstance(obj.get("claims"), list):
        ok, why = _claims_faithful([c for c in obj["claims"] if isinstance(c, dict)], src)
        if not ok:
            return False, why
    return True, ""


_ID_RE = re.compile(r"\b[afpu]\d{1,3}\b")
_GAP_OK = re.compile(r'^(?:[\s"\'\[\]{}(),.:;\-*•]|\b[afpu]\d{1,3}\b|\bevidence ids\b|\bconcept\b|\btext\b|'
                     r'\bclaims\b|\bnull\b|\b[a-z]+(?: [a-z]+){0,4}\b(?=")|\d+[.)](?=\s))*$')


def _claims_faithful(claims: List[dict], src: str) -> Tuple[bool, str]:
    """Typed claims must TILE the response's claim text (no content dropped
    between them, no claim cut inside a token) and each claim's evidence ids
    must be exactly those written in its own unit: the enclosing JSON object
    for JSON-like text, else the line/list item that contains it."""
    nr = _norm_lines(_decode_inplace(src))
    jsonlike = '"text"' in nr or "'text'" in nr
    starts, cursor = [], 0
    for c in claims:
        ct = _norm(str(c.get("text", "")))
        pos = nr.find(ct, cursor)
        while pos >= 0 and ((pos > 0 and nr[pos - 1].isalnum()) or
                            (pos + len(ct) < len(nr) and nr[pos + len(ct)].isalnum() and ct[-1:].isalnum())):
            pos = nr.find(ct, pos + 1)          # token boundaries only
        if pos < 0:
            return False, f"claim not found (in order, at token boundaries): {str(c.get('text'))[:50]!r}"
        starts.append((pos, pos + len(ct)))
        cursor = pos + len(ct)
    for k in range(len(starts) - 1):
        gap = nr[starts[k][1]:starts[k + 1][0]]
        if jsonlike:
            gap = re.sub(r'"[a-z ]+"\s*:', " ", gap)             # JSON keys
            gap = re.sub(r'"[A-Z][A-Z ]+"', " ", gap)               # concept values
            gap = re.sub(r'"(?:[a-z]+(?: [a-z]+){0,5})"', " ", gap)  # short string values (concepts)
        gap = _ID_RE.sub(" ", gap)
        gap = re.sub(r"(?m)(?:^|\s)\d+[.)](?=\s|$)", " ", gap)          # list markers '2.'
        if re.search(r"[a-z]{3,}|\d", re.sub(r"\b(?:evidence ids|concept|text|claims|null|and)\b", " ", gap)):
            return False, f"content between claims {k + 1} and {k + 2} was dropped: {gap.strip()[:40]!r}"
    for k, c in enumerate(claims):
        pos, end = starts[k]
        if jsonlike:
            lo = nr.rfind("{", 0, pos)
            hi = nr.find("}", end)
            unit = nr[lo: hi + 1] if lo >= 0 and hi >= 0 else nr[pos:end]
        else:
            line_lo = nr.rfind("\n", 0, pos) + 1
            line_hi = nr.find("\n", end)
            line_hi = len(nr) if line_hi < 0 else line_hi
            shares = [j for j in range(len(starts)) if j != k and line_lo <= starts[j][0] < line_hi]
            if not shares:
                unit = nr[line_lo:line_hi]          # one claim per line: its whole line
            else:                                    # several claims on one line
                nxt = min([starts[j][0] for j in shares if starts[j][0] > pos] + [line_hi])
                unit = nr[pos:nxt]
        written = set(_ID_RE.findall(unit))
        given = {str(i).lower() for i in (c.get("evidence_ids") or [])}
        if given != written:
            return False, f"evidence ids {sorted(given)} differ from those written with the claim {sorted(written)}"
    return True, ""


_SCHEMA_WORDS = {"explanation", "claims", "text", "evidence_ids", "evidence", "ids", "concept",
                 "accepted", "violations", "type", "detail", "quote", "true", "false", "null",
                 "json", "here", "the", "and", "for", "null"}
VIOLATION_TYPES = ["invented_factor", "sign_reversal", "numeric_or_unit_error",
                   "unsupported_trend", "unsupported_causal_or_action",
                   "omission_of_essential_driver", "other_material_error"]


def _tokens(s: str) -> List[str]:
    toks = re.findall(r"[a-z][a-z']{2,}|[a-z]\d{1,3}\b|-?\d[\d,]*(?:\.\d+)?", _norm(s))
    return [t.replace(",", "") for t in toks if t not in _SCHEMA_WORDS]


def complete(obj: Any, raw: str, kind: str) -> Tuple[bool, str]:
    """The parser must not DROP or ADD content."""
    src = re.sub(r'"[A-Za-z_]+"\s*:', " ", strip_reasoning(raw))      # JSON keys are not content
    src = re.sub(r"(?m)^\s*\d+[.)]\s+", " ", src)                    # list ordinals
    if kind == "producer":
        rw = Counter(_tokens(src))
        if not rw:
            return True, ""
        got = Counter(_tokens(" ".join(_strings(obj)) + " "
                              + " ".join(str(i) for c in (obj.get("claims") or [])
                                         if isinstance(c, dict) for i in (c.get("evidence_ids") or []))
                              if isinstance(obj, dict) else " ".join(_strings(obj))))
        kept = sum(min(n, got[t]) for t, n in rw.items())
        rec = kept / sum(rw.values())
        return (rec >= 0.90, f"parser kept {rec:.0%} of the response's tokens")
    if kind == "judge":
        viol = [v for v in (obj.get("violations") or []) if isinstance(v, dict)]
        low = src.lower()
        named = set()
        for line in low.split("\n"):
            n_types = sum(1 for t in VIOLATION_TYPES if t in line or t.replace("_", " ") in line)
            if n_types >= 3:
                continue                     # a rubric/checklist echo, not a finding
            for t in VIOLATION_TYPES:
                for form in (t, t.replace("_", " ")):
                    for m in re.finditer(re.escape(form), line):
                        pre = re.split(r'["{},]', line[max(0, m.start() - 40):m.start()])[-1]
                        post = line[m.end():m.end() + 25]
                        if re.search(r"\b(no|not|none|without|checked for|check for|free of)\b", pre):
                            continue
                        if re.match(r'^["\s]*[:=-]?\s*["\s]*(none|no|n/a|not found|false|0|absent)\b', post):
                            continue         # 'Sign reversal: none'
                        named.add(t)
        kept = {str(v.get("type", "")).lower().replace(" ", "_") for v in viol}
        if obj.get("accepted") is True and (viol or named):
            return False, "accepting verdict while the response names a violation"
        if obj.get("accepted") is False:
            miss = named - kept
            if miss:
                return False, f"parser dropped violation types {sorted(miss)}"
            if not viol:
                return False, "rejecting verdict with no violation kept"
        return True, ""
    return True, ""


def parse_with_parser_judge(
        raw: str, *, schema_hint: str, validate: Callable[[Any], bool],
        verdict_key: Optional[str] = None, experiment: str = "", item_id: str = "",
        role: str = "", kind: str = "producer",
        finish_reason: Optional[str] = "") -> Tuple[Optional[Any], Dict[str, Any]]:
    """Full contract. Returns (object or None, info); info['method'] in
    direct | repair | parser | truncated | filtered | ambiguous |
    parser_unavailable | unresolved."""
    fin = classify_finish(finish_reason)
    if fin == "truncated":
        return None, {"method": "truncated", "error": "response hit the token limit",
                      "finish": finish_reason}
    if fin == "filtered":
        return None, {"method": "filtered", "error": f"response filtered ({finish_reason})",
                      "finish": finish_reason}
    if fin == "unknown" and looks_truncated(raw):
        return None, {"method": "truncated", "error": f"cut-off JSON with finish reason {finish_reason!r}",
                      "finish": finish_reason}
    obj, how = parse_local(raw, validate, allow_repair=(fin == "clean"))
    if obj is not None:
        return obj, {"method": how, "finish": finish_reason}
    if how == "ambiguous":
        return None, {"method": "ambiguous", "error": "several different JSON answers in one response",
                      "finish": finish_reason}
    import registry
    from gateway import GATEWAY
    info: Dict[str, Any] = {"method": "unresolved", "local": how, "finish": finish_reason,
                            "parser_family": registry.PARSER_FAMILY}
    src = strip_reasoning(raw)
    try:
        r = GATEWAY.complete(
            registry.PARSER_FAMILY,
            [{"role": "system", "content": PARSER_SYS},
             {"role": "user", "content": f"SCHEMA:\n{schema_hint}\n\nRESPONSE TO CONVERT:\n{src[:12000]}"}],
            temperature=0.0, max_tokens=6000, want_json=True,       # > any producer/judge budget
            role=f"parser[{role}]", experiment=experiment, item_id=item_id)
        info.update({"parser_model_id": r.model_id, "parser_call_id": r.call_id,
                     "parser_finish": r.finish_reason})
    except Exception as e:  # noqa: BLE001 - an outage, not a property of the response
        info["method"] = "parser_unavailable"
        info["error"] = f"parser call failed: {type(e).__name__}: {str(e)[:120]}"
        return None, info
    if classify_finish(r.finish_reason) != "clean":
        # the PARSER failed, not the producer/judge: retryable, never a model failure
        info["method"] = "parser_unavailable"
        info["error"] = f"parser output not cleanly finished ({r.finish_reason})"
        return None, info
    pobj, phow = parse_local(r.text, validate, allow_repair=False)
    if pobj is None or phow != "direct":
        info["error"] = "parser output invalid for schema"
        return None, info
    ok, why = faithful(pobj, raw, verdict_key)
    if not ok:
        info["error"] = f"parser output not faithful: {why}"
        return None, info
    ok, why = complete(pobj, raw, kind)
    if not ok:
        info["error"] = f"parser output incomplete: {why}"
        return None, info
    info["method"] = "parser"
    return pobj, info


# ---------------------------------------------------------------------------
# schemas used by the study
# ---------------------------------------------------------------------------
GROUNDED_SCHEMA = '{"explanation": "<the explanation text, verbatim>"}'
TYPED_SCHEMA = ('{"claims": [{"text": "<one sentence, verbatim>", "evidence_ids": ["<id>"], '
                '"concept": "<concept or null>"}]}')
JUDGE_SCHEMA = ('{"accepted": true|false, "violations": [{"type": "<type>", "detail": "<text>", '
                '"quote": "<exact words>", "evidence_ids": ["<id>"]}]}')


def valid_grounded(o) -> bool:
    return isinstance(o, dict) and isinstance(o.get("explanation"), str) and o["explanation"].strip() != ""


def valid_typed(o) -> bool:
    cl = o.get("claims") if isinstance(o, dict) else None
    return isinstance(cl, list) and bool(cl) and all(
        isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip()
        and isinstance(c.get("evidence_ids", []), list)
        and all(isinstance(i, str) for i in c.get("evidence_ids") or [])
        for c in cl)


def valid_judge(o) -> bool:
    return (isinstance(o, dict) and isinstance(o.get("accepted"), bool)
            and isinstance(o.get("violations", []), list)
            and all(isinstance(v, dict) and isinstance(v.get("type"), str)
                    for v in o.get("violations", [])))
