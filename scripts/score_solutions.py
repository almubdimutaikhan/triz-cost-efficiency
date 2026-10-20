#!/usr/bin/env python3
"""
score_solutions.py — Phase 2: an absolute per-solution score, via Jev.

IM-LLM's yield metric needs a threshold on an ABSOLUTE score, not a pairwise
pick (v1's win_rate is a pick, not a score — see paper/PLAN.md for why that
distinction matters). The factorial judge prompt already asks for a 0-10
score, but triz-on-vs-off-agent/judge.py never stored it (raw judge text is
truncated to 400 chars, after which the score was written). Rather than
reuse that truncated pairwise text, this asks Jev directly for an ordered
rating per SOLUTION, independent of any pairing or pick.

Jev's "score" question type takes ordered levels (worst to best) and returns
an expected position normalized to 0-1 — not a raw 0-10 number, but the same
idea: a continuous position on an ordered scale, which is what IM-LLM
thresholds to define a "top idea".

Calls the SAME OpenRouter Decisions endpoint as triz-on-vs-off-agent's
src/judge.py --jev (POST {origin}/api/alpha/decisions, model
typesafe/jev-1.13), reusing that repo's API key / gateway config rather than
duplicating credentials. Disk-cached by a hash of (solution text, rubric),
so a re-run never re-calls the API.

Usage
  python scripts/score_solutions.py --src ../triz-on-vs-off-agent --run factorial_129
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import yaml

JEV_MODEL = "typesafe/jev-1.13"
CACHE_DIR = Path("data/jev_scores")

LEVELS = [
    "Obvious or impractical: a well-known fix already in common use, or not "
    "technically sound as described.",
    "Reasonable but ordinary: solves the problem with a standard, expected "
    "engineering approach.",
    "Solid engineering: technically sound and concrete (a specific mechanism, "
    "not vague advice), but only modestly original.",
    "Inventive: a non-obvious mechanism that is also concrete and feasible — "
    "not just a known fix applied here.",
    "Highly original and elegant: non-obvious, feasible, and solves the "
    "problem with unusually low added complexity, cost or side effects.",
]
INSTRUCTIONS = (
    "Rate this engineering solution's creativity against the problem it answers. Judge "
    "it on its own merits: originality (non-obvious, inventive approach), usefulness "
    "(actually addresses the problem), feasibility (physically and technically "
    "realistic), specificity (a concrete mechanism, not vague advice), and elegance "
    "(low added complexity, cost or side effects), then place it on this scale.")


def api_key_and_base_url(src: Path):
    sys.path.insert(0, str(src / "src"))
    import generate  # the sibling repo's gateway plumbing
    cfg = yaml.safe_load((src / "config.yaml").read_text())
    return generate.api_key(), cfg["gateway_base_url"].rstrip("/")


def call_jev_score(origin: str, key: str, problem: str, solution: str):
    payload = {
        "model": JEV_MODEL,
        "state": json.dumps({"problem": problem, "solution": solution},
                            sort_keys=True, ensure_ascii=False),
        "questions": {"creativity": {"type": "score", "instructions": INSTRUCTIONS,
                                      "criteria": LEVELS}},
    }
    last = None
    for attempt in range(4):
        try:
            r = requests.post(f"{origin}/api/alpha/decisions",
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"},
                              json=payload, timeout=180)
            if r.status_code == 200:
                j = r.json()
                ans = j.get("answers", j)["creativity"]
                pos = ans["score"] if isinstance(ans, dict) else ans
                return float(pos), j.get("usage") or {}, None
            last = f"HTTP {r.status_code}: {r.text[:300]}"
            if r.status_code in (429, 500, 502, 503, 529):
                time.sleep(2 ** attempt)
                continue
            return None, {}, last
        except (requests.RequestException, KeyError, TypeError, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    return None, {}, last


def load_solutions(src: Path, run: str):
    gen_dir = src / "data" / run / "generations"
    out = []
    for f in sorted(gen_dir.glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("error") or not r.get("output"):
            continue
        solution = r.get("display") or r["output"]  # the FINAL SOLUTION-only text judges saw
        out.append({"hash": r["hash"], "case_id": r["case_id"], "mode": r["mode"],
                    "model": r["model"], "problem": r["user"], "solution": solution})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("../triz-on-vs-off-agent"))
    ap.add_argument("--run", default="factorial_129")
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--out", type=Path, default=Path("data/jev_scores.jsonl"))
    ap.add_argument("--limit", type=int, help="only the first N solutions (smoke test)")
    args = ap.parse_args()

    solutions = load_solutions(args.src, args.run)
    if args.limit:
        solutions = solutions[: args.limit]

    # the committed --out file (data/jev_scores.jsonl) IS the reproducible result: if it
    # already has every hash, re-running costs $0 and makes no API calls at all, even with
    # an empty/missing data/jev_scores/ cache dir (e.g. a fresh clone).
    done = set()
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("position") is not None:
                done.add(r["hash"])
    todo = [s for s in solutions if s["hash"] not in done]
    print(f"{len(solutions)} solutions, {len(done)} already in {args.out}, {len(todo)} to score")
    if not todo:
        print("Nothing to do — already fully reproducible from the committed file.")
        return

    key, origin = api_key_and_base_url(args.src.resolve())
    origin = re.sub(r"/api/v1$", "", origin)
    solutions = todo
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def cache_key(sol):
        return hashlib.sha256((sol["solution"] + "|" + json.dumps(LEVELS, sort_keys=True)).encode()).hexdigest()[:16]

    def run_one(sol):
        key_h = cache_key(sol)
        path = CACHE_DIR / f"{key_h}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            rec["_cached"] = True
            return rec
        pos, usage, err = call_jev_score(origin, key, sol["problem"], sol["solution"])
        rec = {"hash": sol["hash"], "case_id": sol["case_id"], "mode": sol["mode"],
              "model": sol["model"], "position": pos, "usage": usage, "error": err}
        path.write_text(json.dumps(rec))
        rec["_cached"] = False
        return rec

    records, n_err, cost = [], 0, 0.0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(run_one, s): s for s in solutions}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            records.append(rec)
            if rec.get("error"):
                n_err += 1
            cost += (rec.get("usage") or {}).get("cost") or 0
            tag = "cache" if rec.get("_cached") else ("ERR " if rec.get("error") else "new ")
            if i % 50 == 0 or i == len(solutions):
                print(f"  [{i}/{len(solutions)}] {tag} last={rec['hash']} pos={rec.get('position')}")

    # merge onto whatever was already in --out, keyed by hash, so a partial re-run (or one
    # limited by --limit) never drops previously-scored solutions
    merged = {}
    if args.out.exists():
        for line in args.out.open(encoding="utf-8"):
            r = json.loads(line)
            merged[r["hash"]] = r
    for r in records:
        merged[r["hash"]] = {k: v for k, v in r.items() if k != "_cached"}
    with args.out.open("w", encoding="utf-8") as fh:
        for r in merged.values():
            fh.write(json.dumps(r) + "\n")
    print(f"\n{len(records)} scored this run, {n_err} errors, ~${cost:.4f} spent -> "
          f"{args.out} ({len(merged)} total)")


if __name__ == "__main__":
    main()
