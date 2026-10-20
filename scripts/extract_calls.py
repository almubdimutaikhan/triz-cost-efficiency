#!/usr/bin/env python3
"""
extract_calls.py — pull one flat row per generation call from the sibling
triz-on-vs-off-agent repo's cache, with exact token counts.

The source repo (data/<run>/generations/*.json) stores the full prompt and
output text but never logged API usage, so token counts here are computed
locally with tiktoken (cl100k_base) — exact for the text, not model-native
for every model family, but applied identically to every arm, which is what
a controlled TRIZ-on vs TRIZ-off comparison needs: the same ruler on both
arms, not the exactly-correct ruler per model.

Zero API calls, zero cost. Source data is read-only; nothing here writes
back to the sibling repo.

Usage
  python scripts/extract_calls.py --src ../triz-on-vs-off-agent --run factorial_129
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import tiktoken

ENC = tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(ENC.encode(text or "", disallowed_special=()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("../triz-on-vs-off-agent"),
                     help="path to the triz-on-vs-off-agent repo")
    ap.add_argument("--run", default="factorial_129")
    ap.add_argument("--out", type=Path, default=Path("data/calls.csv"))
    args = ap.parse_args()

    gen_dir = args.src / "data" / args.run / "generations"
    files = sorted(gen_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No generation records found at {gen_dir}")

    rows = []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        if r.get("error") or not r.get("output"):
            continue
        prompt_tok = n_tokens(r["system"]) + n_tokens(r["user"])
        output_tok = n_tokens(r["output"])
        rows.append({
            "hash": r["hash"], "case_id": r["case_id"], "mode": r["mode"], "model": r["model"],
            "sample_idx": r["sample_idx"], "prompt_tokens": prompt_tok, "output_tokens": output_tok,
            "total_tokens": prompt_tok + output_tok,
            "system_chars": len(r["system"]), "words": len((r.get("display") or r["output"]).split()),
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"{len(rows)} calls -> {args.out}")
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)
    print(f"{'mode':<16}{'n':>5}{'prompt_tok':>12}{'output_tok':>12}{'total_tok':>11}")
    for mode, rs in sorted(by_mode.items()):
        pt = sum(r["prompt_tokens"] for r in rs) / len(rs)
        ot = sum(r["output_tokens"] for r in rs) / len(rs)
        print(f"{mode:<16}{len(rs):>5}{pt:>12.0f}{ot:>12.0f}{pt + ot:>11.0f}")


if __name__ == "__main__":
    main()
