#!/usr/bin/env python3
"""
cost_efficiency.py — does TRIZ priming earn back the extra tokens it costs?

Re-reads the triz-on-vs-off-agent repo's ALREADY-RUN judgements (no new API
calls: results/<run>/judgements.csv = the current cross-family LLM evaluator,
results/<run>/judgements_jev.csv = the Jev decision-model judge added in this
session) and joins them with this repo's token counts (data/calls.csv, from
extract_calls.py) to answer a cost question, not a quality question: is the
longer, TRIZ-primed arm worth what it spends?

Unit of analysis is the CASE (40 problems; see ../triz-on-vs-off-agent's own
invariant), not the judgement: a pair's two presentation orders are collapsed
to one verdict first, and the case-level bootstrap resamples CASES, not
individual judgements, so repeated models/samples per case don't inflate N.

Usage
  python scripts/cost_efficiency.py --src ../triz-on-vs-off-agent --run factorial_129
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

COMPARISONS = [
    ("P_L_TRIZ_ON", "P_L_TRIZ_OFF", "does TRIZ help, at matched (long) prompt length?"),
    ("P_S_TRIZ_ON", "P_S_TRIZ_OFF", "does TRIZ help, at matched (short) prompt length?"),
    ("P_L_TRIZ_ON", "P_S_TRIZ_ON", "does a longer prompt help, with TRIZ already on?"),
    ("P_L_TRIZ_OFF", "P_S_TRIZ_OFF", "does a longer prompt help, with TRIZ off (length alone)?"),
]
BOOTSTRAP = 2000
SEED = 0


def read_csv(path: Path):
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_tokens(path: Path):
    """mode -> {case_id -> [total_tokens, ...]} (one value per generation call)."""
    out = defaultdict(lambda: defaultdict(list))
    for r in read_csv(path):
        out[r["mode"]][r["case_id"]].append(int(r["total_tokens"]))
    return out


def order_consistent_verdicts(judgements_path: Path, arm_a: str, arm_b: str, cross_family_only: bool):
    """One verdict per (judge, pair) where both presentation orders agree.

    Returns {case_id -> [1 if arm_a won, 0 if arm_b won, ...]} (one entry per
    judge x generating-model that was order-consistent on that pair).
    """
    rows = [r for r in read_csv(judgements_path) if not r.get("error")]
    by_pair_judge = defaultdict(list)
    for r in rows:
        pair = r["pair_id"]
        if f"{arm_a}_vs_{arm_b}" not in pair and f"{arm_b}_vs_{arm_a}" not in pair:
            continue
        if cross_family_only and r["judge_model"] == r["gen_model"]:
            continue
        by_pair_judge[(pair, r["judge_model"])].append(r)

    out = defaultdict(list)
    for (pair, judge_model), rs in by_pair_judge.items():
        if len(rs) != 2:
            continue  # need both orders to call it order-consistent
        picks = {r["picked_arm"] for r in rs}
        if len(picks) != 1 or None in picks:
            continue  # order-inconsistent or unparsed: dropped, not counted either way
        case_id = rs[0]["case_id"]
        out[case_id].append(1 if picks == {arm_a} else 0)
    return out


def wilson(p_hat: float, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    d = 1 + z * z / n
    c = (p_hat + z * z / (2 * n)) / d
    h = z * math.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / d
    return round(c - h, 3), round(c + h, 3)


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def case_bootstrap_mean(by_case: dict, rng: random.Random, reps=BOOTSTRAP):
    """95% CI of the case-averaged win rate: resample CASES with replacement,
    and within a resampled case resample its own verdicts too."""
    cases = list(by_case.values())
    draws = []
    for _ in range(reps):
        picked = [rng.choice(cases) for _ in cases]
        means = [sum(rng.choice(c) for _ in c) / len(c) for c in picked]
        draws.append(sum(means) / len(means))
    draws.sort()
    return draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("../triz-on-vs-off-agent"))
    ap.add_argument("--run", default="factorial_129")
    ap.add_argument("--calls", type=Path, default=Path("data/calls.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/cost_efficiency.csv"))
    args = ap.parse_args()

    tokens = load_tokens(args.calls)
    results_dir = args.src / "results" / args.run
    judges = {
        "gemini (current evaluator)": (results_dir / "judgements.csv", True),
        "jev (typesafe/jev-1.13)": (results_dir / "judgements_jev.csv", False),  # Jev has no generator family
    }

    rows = []
    rng = random.Random(SEED)
    for arm_a, arm_b, question in COMPARISONS:
        tok_a = [t for cs in tokens[arm_a].values() for t in cs]
        tok_b = [t for cs in tokens[arm_b].values() for t in cs]
        mean_a, mean_b = sum(tok_a) / len(tok_a), sum(tok_b) / len(tok_b)
        for judge_name, (path, cross_only) in judges.items():
            if not path.exists():
                continue
            by_case = order_consistent_verdicts(path, arm_a, arm_b, cross_only)
            if not by_case:
                continue
            n_cases = len(by_case)
            verdicts = [v for vs in by_case.values() for v in vs]
            wins, losses = sum(verdicts), len(verdicts) - sum(verdicts)
            case_means = [sum(vs) / len(vs) for vs in by_case.values()]
            rate = sum(case_means) / len(case_means)
            lo, hi = case_bootstrap_mean(by_case, rng)
            p = sign_test_p(sum(m > 0.5 for m in case_means), sum(m < 0.5 for m in case_means))
            delta_tok = mean_a - mean_b
            rows.append({
                "comparison": f"{arm_a} vs {arm_b}", "question": question, "judge": judge_name,
                "n_cases": n_cases, "n_verdicts": len(verdicts),
                f"{arm_a}_tokens_mean": round(mean_a), f"{arm_b}_tokens_mean": round(mean_b),
                "delta_tokens": round(delta_tok),
                f"{arm_a}_win_rate": round(rate, 3), "ci95_low": lo, "ci95_high": hi,
                "sign_test_p": round(p, 4), "significant_p05": p < 0.05 and (lo > 0.5 or hi < 0.5),
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} comparison x judge rows -> {args.out}\n")
    for r in rows:
        sig = "significant" if r["significant_p05"] else "not significant"
        win_key = [k for k in r if k.endswith("_win_rate")][0]
        print(f"{r['comparison']:32s} [{r['judge']:26s}] "
              f"win_rate={r[win_key]:.1%} CI[{r['ci95_low']:.1%},{r['ci95_high']:.1%}] "
              f"p={r['sign_test_p']:.3f} ({sig}, n={r['n_cases']} cases)  "
              f"Δtokens={r['delta_tokens']:+d}")


if __name__ == "__main__":
    main()
