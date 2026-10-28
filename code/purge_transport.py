"""
Remove rows whose outcome was a provider outage (transport error or wall-clock timeout) so that the
resumable runners redo exactly those items. A backup of every touched file is written first.
Outages are operational failures of a route, not verification outcomes; each purge is logged.

    python purge_transport.py <file.jsonl> [...]
"""
import json
import shutil
import sys
import time

MARKERS = ("transport:", "FamilyExhausted", "wall_clock_timeout", "unparseable: transport")

for p in sys.argv[1:]:
    rows = [json.loads(l) for l in open(p)]
    def bad(r):
        s = json.dumps({k: r.get(k) for k in ("error", "reason")}) + json.dumps((r.get("detail") or {}).get("reason", ""))
        return any(m in s for m in MARKERS)
    keep = [r for r in rows if not bad(r)]
    if len(keep) == len(rows):
        print(f"{p}: nothing to purge")
        continue
    shutil.copy(p, p + f".bak_{time.strftime('%H%M%S')}")
    with open(p, "w") as f:
        for r in keep:
            f.write(json.dumps(r, default=str) + "\n")
    with open(p + ".purge_log", "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "removed": len(rows) - len(keep),
                            "items": [r["item_id"] for r in rows if bad(r)]}) + "\n")
    print(f"{p}: removed {len(rows) - len(keep)} outage rows")
