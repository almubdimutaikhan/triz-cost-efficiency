#!/usr/bin/env python3
"""
yield_analysis.py — Phase 2: IM-LLM's actual metric, top solutions per 100k tokens.

Unlike v1 (cost_efficiency.py, which compares token cost against a pairwise
WIN RATE), this thresholds Jev's absolute per-solution score (score_solutions.py)
and counts solutions above it — the same construction as IM-LLM's tail_charts.py:
yield = all top solutions / all tokens, a slope, not a saturating probability.
Case-clustered bootstrap (calls resampled within case) and an exact sign test
across cases, matching tail_charts.py and v1's cost_efficiency.py.

Usage
  python scripts/yield_analysis.py --threshold 2.2
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

BOOTSTRAP = 2000
SEED = 0
MATCHED_PAIRS = [("P_L_TRIZ_ON", "P_L_TRIZ_OFF"), ("P_S_TRIZ_ON", "P_S_TRIZ_OFF")]


def load(calls_path: Path, scores_path: Path):
    tokens = {}
    for r in csv.DictReader(calls_path.open(encoding="utf-8")):
        tokens[r["hash"]] = int(r["total_tokens"])
    calls = []
    for line in scores_path.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("position") is None or r["hash"] not in tokens:
            continue
        calls.append({"hash": r["hash"], "case_id": r["case_id"], "mode": r["mode"],
                      "model": r["model"], "position": r["position"], "tokens": tokens[r["hash"]]})
    return calls


def by_mode_case(calls, mode):
    out = defaultdict(list)
    for c in calls:
        if c["mode"] == mode:
            out[c["case_id"]].append(c)
    return out


def top_rate(per_case, th):
    """Case-balanced top-idea rate: mean over cases of (that case's share of top solutions)."""
    vals = [sum(c["position"] >= th for c in cs) / len(cs) for cs in per_case.values()]
    return sum(vals) / len(vals)


def yield_per_100k(per_case, th):
    hits = sum(c["position"] >= th for cs in per_case.values() for c in cs)
    toks = sum(c["tokens"] for cs in per_case.values() for c in cs)
    return 1e5 * hits / toks if toks else float("nan")


def bootstrap(per_case, th, rng, reps=BOOTSTRAP):
    rate_draws, yield_draws = [], []
    for _ in range(reps):
        res = {k: [rng.choice(cs) for _ in cs] for k, cs in per_case.items()}
        rate_draws.append(top_rate(res, th))
        yield_draws.append(yield_per_100k(res, th))
    rate_draws.sort(); yield_draws.sort()
    lo, hi = int(0.025 * reps), int(0.975 * reps) - 1
    return (rate_draws[lo], rate_draws[hi]), (yield_draws[lo], yield_draws[hi]), yield_draws


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calls", type=Path, default=Path("data/calls.csv"))
    ap.add_argument("--scores", type=Path, default=Path("data/jev_scores.jsonl"))
    ap.add_argument("--threshold", type=float, default=2.2,
                     help="Jev expected-position cut for a 'top solution' (default 2.2 of 0-4, "
                          "the nearest round value to the top ~12%% of all solutions)")
    ap.add_argument("--out", type=Path, default=Path("results/yield.csv"))
    args = ap.parse_args()

    calls = load(args.calls, args.scores)
    all_pos = [c["position"] for c in calls]
    kept = sum(p >= args.threshold for p in all_pos) / len(all_pos)
    print(f"threshold {args.threshold}: keeps {sum(p >= args.threshold for p in all_pos)} of "
          f"{len(all_pos)} solutions ({kept:.1%})\n")

    modes = sorted({c["mode"] for c in calls})
    rng = random.Random(SEED)
    rows, by_mode = [], {}
    for mode in modes:
        per_case = by_mode_case(calls, mode)
        by_mode[mode] = per_case
        rate = top_rate(per_case, args.threshold)
        yld = yield_per_100k(per_case, args.threshold)
        (rlo, rhi), (ylo, yhi), ydraws = bootstrap(per_case, args.threshold, rng)
        n = sum(len(cs) for cs in per_case.values())
        hits = sum(c["position"] >= args.threshold for cs in per_case.values() for c in cs)
        toks = sum(c["tokens"] for cs in per_case.values() for c in cs)
        rows.append({"mode": mode, "cases": len(per_case), "calls": n, "top_solutions": hits,
                    "top_rate": round(rate, 4), "rate_ci_lo": round(rlo, 4), "rate_ci_hi": round(rhi, 4),
                    "tokens_total": toks, "yield_per_100k": round(yld, 3),
                    "yield_ci_lo": round(ylo, 3), "yield_ci_hi": round(yhi, 3), "_draws": ydraws})

    print(f"{'mode':<16}{'calls':>6}{'top':>5}{'top rate':>22}{'yield/100k tok':>22}")
    for r in rows:
        rate_s = f"{r['top_rate']:.1%} [{r['rate_ci_lo']:.1%},{r['rate_ci_hi']:.1%}]"
        yield_s = f"{r['yield_per_100k']:.2f} [{r['yield_ci_lo']:.2f},{r['yield_ci_hi']:.2f}]"
        print(f"{r['mode']:<16}{r['calls']:>6}{r['top_solutions']:>5}{rate_s:>22}{yield_s:>22}")

    # H1: at matched prompt length, is TRIZ-on's yield LOWER than TRIZ-off's?
    print("\nH1 (paper/PLAN.md): TRIZ-on yield < TRIZ-off yield, at matched length")
    by_mode_row = {r["mode"]: r for r in rows}
    h1_rows = []
    for on, off in MATCHED_PAIRS:
        r_on, r_off = by_mode_row[on], by_mode_row[off]
        p_on_higher = sum(a > b for a, b in zip(r_on["_draws"], r_off["_draws"])) / len(r_on["_draws"])
        # per-case sign test: for each case (TRIZ-on vs off), whose yield is higher?
        cases = sorted(set(by_mode[on]) & set(by_mode[off]))
        on_wins = off_wins = 0
        for c in cases:
            y_on = yield_per_100k({c: by_mode[on][c]}, args.threshold)
            y_off = yield_per_100k({c: by_mode[off][c]}, args.threshold)
            if y_on > y_off:
                on_wins += 1
            elif y_off > y_on:
                off_wins += 1
        p = sign_test_p(on_wins, off_wins)
        # require the pooled (token-weighted) bootstrap AND the per-case sign test to agree
        # before calling it supported/rejected — a pooled-only signal is token-weighted, so a
        # few token-heavy cases can swing it even when most cases don't individually agree.
        pooled_says_lower, pooled_says_higher = p_on_higher < 0.025, p_on_higher > 0.975
        case_says_lower, case_says_higher = (p < 0.05 and on_wins < off_wins), (p < 0.05 and on_wins > off_wins)
        if pooled_says_lower and (case_says_lower or p >= 0.05):
            verdict = "H1 SUPPORTED" if case_says_lower else "pooled-only signal (not confirmed per-case)"
        elif pooled_says_higher:
            verdict = "H1 REJECTED" if case_says_higher else "pooled-only signal, opposite direction"
        else:
            verdict = "inconclusive"
        print(f"  {on} vs {off}: P(TRIZ-on higher)={p_on_higher:.3f} | "
              f"per-case {on_wins}-{off_wins} (of {len(cases)}), sign-test p={p:.4f} -> {verdict}")
        h1_rows.append({"pair": f"{on}_vs_{off}", "p_triz_on_higher_yield": round(p_on_higher, 4),
                        "cases_triz_on_higher": on_wins, "cases_triz_off_higher": off_wins,
                        "n_cases": len(cases), "sign_test_p": round(p, 4)})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        keys = [k for k in rows[0] if k != "_draws"]
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows({k: r[k] for k in keys} for r in rows)
    h1_path = args.out.with_name("yield_h1_test.csv")
    with h1_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(h1_rows[0].keys()))
        w.writeheader()
        w.writerows(h1_rows)
    print(f"\n-> {args.out}\n-> {h1_path}")


if __name__ == "__main__":
    main()
