#!/usr/bin/env python3
"""
charts.py — the three figures embedded in README.md: yield per arm, win rate
per arm, and yield by generator model. Same construction as IM-LLM's
tail_charts.py / analyze.py (case-clustered bootstrap, 95% CI, consistent
colour per condition across every chart), reading only this repo's own
already-computed CSVs/JSONL — no new API calls, no sibling-repo dependency.

Usage
  python scripts/charts.py
"""
from __future__ import annotations

import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

OUT = Path("results/charts")
BOOTSTRAP = 2000
SEED = 0

# One colour per TRIZ setting, used identically in every chart; one per generator model.
# Chosen to be distinguishable without relying on a red/green pair.
TRIZ_COLOR = {"ON": "#2a6f97", "OFF": "#e67e22"}  # blue = TRIZ on, orange = TRIZ off
MODEL_COLOR = {
    "openai/gpt-4o": "#2a6f97",
    "anthropic/claude-sonnet-4.5": "#8e44ad",
    "deepseek/deepseek-chat-v3.1": "#c0392b",
    "google/gemini-2.5-flash": "#d4a017",
}
MODEL_LABEL = {
    "openai/gpt-4o": "GPT-4o",
    "anthropic/claude-sonnet-4.5": "Claude Sonnet 4.5",
    "deepseek/deepseek-chat-v3.1": "DeepSeek V3.1",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
}
ARM_LABEL = {
    "P_L_TRIZ_OFF": "Long prompt · TRIZ off", "P_L_TRIZ_ON": "Long prompt · TRIZ on",
    "P_S_TRIZ_OFF": "Short prompt · TRIZ off", "P_S_TRIZ_ON": "Short prompt · TRIZ on",
}
INK, GRID, MUTED = "#1a1a1a", "#dddddd", "#888888"


def read_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def bootstrap_yield_ci(per_case, th, rng, reps=BOOTSTRAP):
    draws = []
    for _ in range(reps):
        res = {k: [rng.choice(cs) for _ in cs] for k, cs in per_case.items()}
        hits = sum(c["position"] >= th for cs in res.values() for c in cs)
        toks = sum(c["tokens"] for cs in res.values() for c in cs)
        draws.append(1e5 * hits / toks if toks else float("nan"))
    draws.sort()
    return draws[int(0.025 * reps)], draws[int(0.975 * reps) - 1]


def finish(fig, ax, name, caption, left=0.28):
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=left, bottom=0.3 + 0.045 * caption.count("\n"), top=0.86)
    fig.text(0.01, 0.01, caption, fontsize=7.5, color=MUTED, va="bottom")
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=180)
    plt.close(fig)
    print(f"  {OUT / name}")


# ── Chart 1: yield per arm, with 95% CI ──────────────────────────────
def chart_yield_by_arm():
    rows = read_csv(Path("results/yield.csv"))
    order = ["P_L_TRIZ_OFF", "P_L_TRIZ_ON", "P_S_TRIZ_OFF", "P_S_TRIZ_ON"]
    rows = sorted(rows, key=lambda r: order.index(r["mode"]))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        on_off = "ON" if r["mode"].endswith("TRIZ_ON") else "OFF"
        col = TRIZ_COLOR[on_off]
        lo, hi, mid = float(r["yield_ci_lo"]), float(r["yield_ci_hi"]), float(r["yield_per_100k"])
        ax.plot([lo, hi], [y, y], color=col, linewidth=2.4, solid_capstyle="round")
        ax.scatter([mid], [y], s=80, color=col, zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(hi + 0.4, y, f"{mid:.1f}  [{lo:.1f}, {hi:.1f}]", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(ys, [ARM_LABEL[r["mode"]] for r in rows])
    ax.set_xlabel("top solutions per 100k tokens (threshold 2.2, ≈ top 12% of all solutions)")
    ax.set_title("Yield per arm, with 95% CI (case-clustered bootstrap, n=40 cases)", loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(0, max(float(r["yield_ci_hi"]) for r in rows) * 1.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    handles = [plt.Line2D([], [], color=TRIZ_COLOR[k], marker="o", linestyle="none", markersize=7,
                          label=f"TRIZ {k.lower()}") for k in ("ON", "OFF")]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)
    finish(fig, ax, "yield_by_arm.png",
          "Blue = TRIZ on, orange = TRIZ off · dot = observed yield, bar = 95% CI\n"
          "Source: results/yield.csv (scripts/yield_analysis.py)", left=0.3)


# ── Chart 2: win rate per arm (v1), with 95% CI ──────────────────────
def chart_win_rate():
    rows = [r for r in read_csv(Path("results/cost_efficiency.csv"))
           if r["comparison"].split(" vs ")[1].split("_TRIZ_")[0] == r["comparison"].split(" vs ")[0].split("_TRIZ_")[0]]
    # keep only the TRIZ-on vs TRIZ-off rows (matched length), both judges
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ys = list(range(len(rows)))[::-1]
    ax.axvline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=0)
    ax.text(0.5, len(rows) - 0.15, " no difference", color=MUTED, fontsize=8, va="bottom")
    ax.set_ylim(-0.8, len(rows) - 0.1)
    win_key = [k for k in rows[0] if k.endswith("_win_rate")][0] if rows else None
    for y, r in zip(ys, rows):
        length = "Long" if "P_L" in r["comparison"] else "Short"
        win_key_r = [k for k in r if k.endswith("_win_rate") and r[k]][0]
        rate, lo, hi = float(r[win_key_r]), float(r["ci95_low"]), float(r["ci95_high"])
        col = TRIZ_COLOR["ON"]
        ax.plot([lo, hi], [y, y], color=col, linewidth=2.4, solid_capstyle="round")
        ax.scatter([rate], [y], s=80, color=col, zorder=3, edgecolor="white", linewidth=0.8)
        ax.text(hi + 0.02, y, f"{rate:.1%}  [{lo:.1%}, {hi:.1%}]", va="center", fontsize=8.5, color=INK)
        ax.text(-0.02, y, f"{length} · {r['judge'].split(' (')[0]}", va="center", ha="right", fontsize=9, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("TRIZ-on win rate vs TRIZ-off (pairwise, order-consistent, case-clustered)")
    ax.set_title("Win rate: TRIZ-on vs TRIZ-off, by judge (v1)", loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    finish(fig, ax, "win_rate_by_arm.png",
          "Every CI straddles 50% (no win-rate advantage detected) · source: results/cost_efficiency.csv",
          left=0.22)


# ── Chart 3: yield by generator model ────────────────────────────────
def chart_yield_by_model():
    tokens = {r["hash"]: int(r["total_tokens"]) for r in read_csv(Path("data/calls.csv"))}
    calls = []
    for line in Path("data/jev_scores.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("position") is None or r["hash"] not in tokens:
            continue
        calls.append({**r, "tokens": tokens[r["hash"]],
                      "triz": "ON" if r["mode"].endswith("TRIZ_ON") else "OFF"})

    th = 2.2
    rng = random.Random(SEED)
    models = sorted({c["model"] for c in calls}, key=lambda m: MODEL_LABEL.get(m, m))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rows_out = []
    y = 0
    yticks, yticklabels = [], []
    for model in models:
        for on_off in ("ON", "OFF"):
            cs = [c for c in calls if c["model"] == model and c["triz"] == on_off]
            per_case = defaultdict(list)
            for c in cs:
                per_case[c["case_id"]].append(c)
            hits = sum(c["position"] >= th for c in cs)
            toks = sum(c["tokens"] for c in cs)
            mid = 1e5 * hits / toks if toks else float("nan")
            lo, hi = bootstrap_yield_ci(per_case, th, rng)
            col = MODEL_COLOR[model]
            marker = "o" if on_off == "ON" else "D"
            fill = col if on_off == "ON" else "white"
            ax.plot([lo, hi], [y, y], color=col, linewidth=2.2, solid_capstyle="round", alpha=0.9)
            ax.scatter([mid], [y], s=75, facecolor=fill, edgecolor=col, linewidth=1.6, zorder=3)
            ax.text(hi + 0.6, y, f"{mid:.1f}  [{lo:.1f}, {hi:.1f}]  (n={len(cs)})",
                   va="center", fontsize=8, color=INK)
            yticks.append(y)
            yticklabels.append(f"{MODEL_LABEL[model]} · TRIZ {on_off.lower()}")
            rows_out.append({"model": model, "triz": on_off, "n": len(cs), "hits": hits,
                            "yield_per_100k": round(mid, 2), "ci_lo": round(lo, 2), "ci_hi": round(hi, 2)})
            y -= 1
        y -= 0.4  # gap between models
    ax.set_yticks(yticks, yticklabels, fontsize=8.5)
    for lab, row in zip(ax.get_yticklabels(), rows_out):
        lab.set_color(MODEL_COLOR[row["model"]])
    ax.set_xlabel("top solutions per 100k tokens, pooled over both prompt lengths (threshold 2.2)")
    ax.set_title("Yield by generator model: filled = TRIZ on, open = TRIZ off", loc="left", fontsize=11)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    finish(fig, ax, "yield_by_model.png",
          "Colour = generator model (the model that WROTE the solution, not the judge) · dot = observed "
          "yield, bar = 95% CI\nn = solutions behind that bar (≤80: 40 cases × 2 prompt lengths, fewer "
          "if a case/model/mode cell failed to parse)", left=0.26)

    with Path("results/yield_by_model.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"  results/yield_by_model.csv")


def main():
    chart_yield_by_arm()
    chart_win_rate()
    chart_yield_by_model()


if __name__ == "__main__":
    main()
