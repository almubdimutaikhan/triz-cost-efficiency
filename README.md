# triz-cost-efficiency

**Does the TRIZ system prompt earn back the extra tokens it costs?**

A re-analysis of the [`triz-on-vs-off-agent`](../triz-on-vs-off-agent) factorial study
(`run: factorial_129`), reframed around **token cost** instead of creativity rating — the
same move [`orioli/IM-LLM`](https://github.com/orioli/IM-LLM) makes for product-graph
context (`top ideas per 100k tokens`, not "is the full graph judged better"). This repo
copies that paper's structure and statistical rigor (case-clustered bootstrap, exact sign
tests, blind cross-family judging) and applies it to a different question: for TRIZ
priming, is the token premium worth it?

This README **is the plan**, not a finished paper. Section "Status" says exactly what has
been run (v1, below — zero new API spend) and what hasn't (Phase 2).

---

## The question

`triz-on-vs-off-agent`'s factorial design already isolates two independent variables:

|                 | TRIZ on           | TRIZ off           |
|-----------------|--------------------|--------------------|
| **Long prompt** | `P_L_TRIZ_ON`      | `P_L_TRIZ_OFF`     |
| **Short prompt**| `P_S_TRIZ_ON`      | `P_S_TRIZ_OFF`     |

`P_L_TRIZ_OFF` is deliberately length- and structure-matched to `P_L_TRIZ_ON` (see that
repo's `config.yaml`), so a TRIZ-on vs TRIZ-off comparison at matched length isolates
**content** (the 40 inventive principles) from **length** as the thing that costs tokens.
That repo's own headline question is "does TRIZ-on win more 2AFC judgements?" (yes, ~55%,
per its `RESEARCH_REPORT.md`). This repo asks a different, narrower question:

> **RQ.** Holding prompt length fixed, does TRIZ priming win enough judgements per extra
> token spent to be worth using — or does it just cost more for a result indistinguishable
> from TRIZ-off?

This is IM-LLM's move exactly: IM-LLM found the full graph wins more *whole-set*
comparisons (64-65%) but is **no more efficient** per token than a relevant slice — "more
context buys better average sets, not more great ideas." Here the analogous question is
whether TRIZ's extra reasoning tokens buy a real win-rate edge, or just a longer answer.

## What's different from IM-LLM, and why the metric has to change

IM-LLM's headline metric, **yield** (top ideas per 100k tokens), needs two things this
study doesn't have:

1. **Multiple outputs per call**, so a count can accumulate (IM-LLM asks for 5 ideas per
   call; TRIZ-on/off produces **one** solution per call).
2. **An absolute per-output quality score with a fixed threshold**, so "top idea" is
   well-defined. The factorial evaluator here (`prompts/factorial/E_TRIZ_OFF.txt`) *does*
   ask the judge for `SCORE A: 0-10` / `SCORE B: 0-10` before its final pick — but
   `judge.py` only ever cached the **parsed A/B pick**, truncating the judge's raw text to
   400 characters (the scores sit after that cut, at the end of the response). So absolute
   scores were never recoverable from the existing judgements, only cheaply recoverable by
   construction (see Phase 2, since the scores were asked for — they just weren't stored).

So **v1** below reports the IM-LLM-style quantities that *are* available now, at zero extra
cost — token cost per arm, and the existing pairwise win rate per arm, side by side (IM-LLM's
Figure 5 "context ladder" framing: cost vs quality, not a saturating-count yield). **Phase 2**
(not yet run) adds the absolute scoring needed for a true threshold-based yield metric.

## Status

- ✅ **v1 — cost vs. win-rate, zero new API calls.** Done. See `results/cost_efficiency.csv`
  and Results below. Reuses `triz-on-vs-off-agent`'s already-run judgements (the current
  cross-family LLM evaluator, and the Jev decision-model judge added in that project's most
  recent session) and computes exact token counts locally with `tiktoken` on text already
  sitting in that repo's cache.
- ⏳ **Phase 2 — absolute per-solution scoring, for a true yield metric.** Not started. See
  "Phase 2" below. Trivial cost (Jev judged 640 pairs for $0.025 in the sibling project;
  absolute scoring is the same order of magnitude).
- ⏳ **Phase 3 — write-up.** Not started; depends on Phase 2's result.

## Pipeline

```
 ../triz-on-vs-off-agent (read-only; a sibling clone of that repo is required)
   data/factorial_129/generations/*.json   — full prompt + output text, no token counts logged
   results/factorial_129/judgements*.csv   — already-run pairwise A/B verdicts (this repo adds nothing new)
        │
        ▼ scripts/extract_calls.py   (tiktoken, cl100k_base — same ruler on every arm, zero cost)
   data/calls.csv                          — one row per call: prompt/output/total tokens
        │
        ▼ scripts/cost_efficiency.py (case-clustered bootstrap, Wilson CI, exact sign test — same
        │                             stats recipe as both IM-LLM's tail_charts.py and
        │                             triz-on-vs-off-agent's src/judge_report.py)
   results/cost_efficiency.csv             — token cost x win rate x significance, per comparison x judge
```

### Reproduce

```bash
pip install -r requirements.txt   # just tiktoken
# requires ../triz-on-vs-off-agent cloned alongside this repo (its generations + judgements
# are read, not duplicated here — see "Why no vendored data" below)
python scripts/extract_calls.py --src ../triz-on-vs-off-agent --run factorial_129
python scripts/cost_efficiency.py --src ../triz-on-vs-off-agent --run factorial_129
```

No API key needed for v1 — everything it reads was already generated and judged by the
sibling project; token counting is local and free.

**Why no vendored data copy.** IM-LLM ships its `runs/`/`results/` directly so the paper
rebuilds fully offline with no sibling dependency. This repo was set up to do the same, but
bulk-copying the sibling repo's generation/judgement files into a second repo destined for
GitHub is exactly the shape of an unintended data-exfiltration pattern, so that step was
deliberately not taken. The dependency on a sibling clone is the honest tradeoff instead.

## Results (v1)

From `results/cost_efficiency.csv`. Unit of analysis is the **case** (40 problems that have
TRIZ-on/off pairs — see the sibling repo's `JEV_TRIZ_REPORT.md`), not the judgement: each
pair's two presentation orders are collapsed to one verdict first (order-inconsistent pairs
dropped), then the win rate is averaged per case before pooling across cases, and the 95%
CI comes from 2,000 case-resampled bootstrap draws.

| Comparison | Judge | Δ tokens | Win rate | 95% CI | Sign-test p | Significant? |
|---|---|---|---|---|---|---|
| TRIZ-on vs off, **long** prompt | Gemini (current evaluator) | **+85** | 53.8% | [40.6, 68.4] | 0.711 | No |
| TRIZ-on vs off, **long** prompt | Jev | **+85** | 62.1% | [47.7, 75.8] | 0.108 | No |
| TRIZ-on vs off, **short** prompt | Gemini | **+46** | 53.0% | [38.9, 66.2] | 0.728 | No |
| TRIZ-on vs off, **short** prompt | Jev | **+46** | 49.3% | [35.8, 63.7] | 1.000 | No |
| Long vs short, TRIZ on | Gemini | +704 | 47.0% | [34.2, 59.8] | 1.000 | No |
| Long vs short, TRIZ off | Gemini | +665 | 47.1% | [33.3, 60.8] | 1.000 | No |

**Reading this.**

1. **TRIZ priming has a real, measured token cost** — +85 tokens/call at matched long-prompt
   length, +46 at matched short-prompt length (both judges agree on these; token counts don't
   depend on the judge). The premium is almost entirely **output** tokens: the length-matched
   prompts keep input tokens close (674 vs 665 long; 240 vs 237 short — see `data/calls.csv`),
   so the cost is the model reasoning more before writing `FINAL SOLUTION:`, not reading more.
2. **No win-rate advantage clears significance at n=40 cases**, by either judge. Gemini's
   point estimates sit just above 50% (53-54%); Jev's are more scattered (62% long, 49%
   short) but every CI still straddles 50%. This doesn't contradict the sibling project's
   own ~55% headline (`RESEARCH_REPORT.md`) — it's the *same* result read through a
   case-clustered lens instead of a pooled-judgement one, which is a stricter, smaller-N test
   (40 cases vs hundreds of judgements), so a result that looked like a real effect pooled
   can legitimately not clear this bar.
3. **Prompt length alone costs far more than TRIZ content does** (+665 to +704 tokens) for
   a win rate indistinguishable from 47% — i.e. worse than a coin flip, not better. This
   echoes IM-LLM's own finding that more context is not free even when it's irrelevant to
   the question.

**Caveat on Jev.** Jev's n drops to 33-37 cases (vs Gemini's 39-40) because it has a strong
position bias (only ~52% of its pairs are order-consistent, per the sibling repo's own
`JEV_TRIZ_REPORT.md`), so its intervals are wider and its point estimates noisier. Read the
Gemini row as primary and the Jev row as a cheap, partially-independent cross-check.

## Phase 2 (planned, not run)

To get an IM-LLM-style **yield** (e.g. "wins per 1,000 extra tokens," or a true top-solution
rate against an absolute threshold) rather than a bare win-rate-vs-tokens table, the judge
needs to emit — and this repo needs to **store**, not just parse — an absolute per-solution
score, the way IM-LLM's `top_ideas.py` does. Concretely:

1. Re-prompt (cheap: Jev judged all 640 pairs in the sibling project for $0.025; an
   absolute-scoring pass over ~320 unique solutions is the same order of magnitude) for a
   **0-10 score per solution**, independent of any pairing — not reusing the existing pairwise
   judgements, whose raw text was truncated before the scores were ever written down.
2. Define a top-solution threshold the same principled way IM-LLM did (§3.6 of that paper):
   pick the step closest to the top ~10-12% of all solutions, report the exact share it
   keeps, and check a stricter cut as a robustness check.
3. Compute yield = top solutions / 100k tokens per arm, with the same case-clustered
   bootstrap as `tail_charts.py` uses, and the same arm-vs-arm sign test across cases.
4. Pre-register the comparison before running it (as IM-LLM's persona follow-up and this
   project's own `M6_50ideas/` do): the prediction, stated now, is that TRIZ-on's yield will
   be *lower* than TRIZ-off's at matched length, mirroring v1's win-rate-vs-cost pattern —
   this is written down before Phase 2 runs so the result isn't fit after the fact.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/extract_calls.py` | token counts per call, from the sibling repo's cached text (zero API cost) |
| `scripts/cost_efficiency.py` | joins token counts with existing judgements; cost-vs-win-rate stats |
| `data/calls.csv` | one row per generation call: prompt/output/total tokens (tiktoken, cl100k_base) |
| `results/cost_efficiency.csv` | the table in "Results" above |
| `paper/PLAN.md` | the pre-registered Phase 2 design, in more detail than this README |

## Provenance

All source data (problems, prompts, generations, judgements) is `triz-on-vs-off-agent`'s;
see that repo's `AGENTS.md` for the casebase and generation pipeline, and its
`JEV_TRIZ_REPORT.md` for the Jev judge this repo's Jev row reuses. This repo adds no new
generations and no new judgements in v1 — only token counting and a different statistical
cut through data that already existed.
