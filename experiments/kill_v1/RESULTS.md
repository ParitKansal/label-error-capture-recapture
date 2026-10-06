# Kill experiment v1: results

Run on Google Colab on 2026-10-06. Judged against `PREREG.md` (blob `18572277ccc225ea15db963898f9723bb729be86`,
recorded by the analysis notebook before it read any expert label). Raw outputs are in `data/`;
`data/analysis_summary.json` holds every number below.

## Verdict: NO-GO

The pre-registered rule (PREREG §8) says NO-GO if the union of detectors already finds at least 90% of the
MMLU-Redux key errors. It found 171 of 181 (94.5%). Conditions (c) and (d) fail; (a) and (b) pass.

| | Condition | Result | |
|---|---|---|---|
| (a) | Coverage over ≥ 200 injection replicates ≥ 85% | 100.0% over 200 replicates | PASS |
| (b) | Pooled Redux 95% interval covers 181 | estimate 181.9, interval [171.0, 218.3] | PASS |
| (c) | Union misses ≥ 20% of the 181 Redux key errors | union finds 171, misses 5.5% | FAIL |
| (d) | \|estimate − truth\| ≤ ½ \|union − truth\| on Redux and on average over replicates | Redux 0.9 vs 5.0 (passes); replicates 2.0 vs 0.9 (fails) | FAIL |

## What happened, in plain words

The counting method worked. On the real benchmark it estimated 181.9 key errors against an expert count of
181, and on planted errors its 95% interval contained the true count in all 200 replicates. The fit was
acceptable (goodness-of-fit p = 0.33 on Redux, median 0.41 over replicates), and no fit was unstable.

But the experiment could not show that the method is useful, because the detectors left almost nothing for it
to estimate. The four 2024-era judges said "this answer key is wrong" far too often:

| Detector | Items flagged (of 5,700) | Key errors caught (of 181) | Recall | Precision |
|---|---|---|---|---|
| D1 Llama-3-8B | 2,672 (47%) | 129 | 71.3% | 4.9% |
| D2 Mistral-7B | 2,414 (42%) | 133 | 73.5% | 5.6% |
| D3 Qwen2-7B | 3,646 (64%) | 157 | 86.7% | 4.3% |
| D4 Phi-3-medium | 1,999 (35%) | 134 | 74.0% | 6.8% |
| D5 confident learning | 925 (16%) | 88 | 48.6% | 9.6% |

Together they flagged 4,145 of 5,700 items (73% of the benchmark) to catch 171 errors. High recall bought this
way is useless for an audit: a person would have to check almost the whole benchmark. On the injected errors
(random key swaps, which are easy to spot) the union caught 264.3 of 266 on average, so the raw count was
already almost exact and condition (d) could not be met there.

This is the same lesson as the previous project, from the other side: false positives, not the estimator,
decide whether the setup is informative.

## Other pre-registered readings

- **Errors no detector found (Redux):** 10.9, interval [0.0, 47.3].
- **Audit cost:** checking the 4,145 flagged items finds 171 key errors.
- **Supercategories (secondary):** the interval covers the expert count in all four (STEM 59, humanities 43,
  other 63, social sciences 16).
- **Secondary estimators (Redux):** Mh jackknife 183.0 [177.4, 214.1]; Chao lower bound 179.8 [173.6, 200.7].
  The pairwise Chapman estimates (147–166) fall below the union, as expected when detectors are positively
  dependent. Their printed intervals show a lower bound above the upper bound because the lower end is floored
  at the union count; this is a display bug in `crc.chapman`, it does not affect any GO condition.
- **Selected interactions (Redux):** seven two-way terms, including D1:D5 and D4:D5, so confident learning is
  not independent of the judges, as §6 anticipated.
- **Unparsed rates:** at most 0.6% (Mistral). Only 3 of 67,840 judge answers were read by the parser's
  fallback rather than the `VERDICT:` line, so parsing did not influence the result.

## Exploratory (not part of the verdict)

**Position memorisation (§9a).** Share of the 181 key errors each judge calls correct, original vs shuffled
option order: Llama 30.9% vs 28.7%, Mistral 30.4% vs 26.5%, Qwen2 10.5% vs 13.3%, Phi-3 27.1% vs 26.0%.
The differences are within noise for n = 181 and go both ways. No clear evidence of position memorisation in
these models; content memorisation is not tested by this check.

**Modern judges (§9b).** `google/gemma-4-31B-it-qat-w4a16-ct` ran on the Redux pass: it flagged 730 items
(12.8%), caught 108 of the 181 key errors (recall 59.7%, precision 14.9%) and so missed 73. A newer model is
three times more precise than the old judges and misses about 40% of the errors, which is the regime where an
estimate of the missed errors matters. Contamination can only make this model look better, so the miss rate is
if anything understated. `Qwen/Qwen3.8-27B-FP8` was skipped by the smoke check (9.6% unparsed): it is a
thinking model and spends its 24 output tokens reasoning before the verdict.

## Deviations from the plan

All environment-only; none touched `PREREG.md` or any threshold.

1. Judge notebooks: Colab ships transformers 5, which vLLM 0.11.0 cannot load; pinned transformers below 5.
2. `judge_llama3`: the token was not exported to vLLM's engine process, so the gated weights failed to load;
   the setup cell now exports `HF_TOKEN`.
3. `modern_arm`: moved to vLLM 0.31.0, which supports the 2026 architectures.
4. The analysis notebook as first built lacked the loading, D5, Redux and GO cells; they were written to the
   letter of §3–§8 before the notebook read any label, and tested end to end on synthetic outputs.
5. The analysis ran on a TPU runtime for its 44 CPU cores; the result does not depend on the runtime (fixed seeds).

## What a next version would test

The method needs detectors that miss a meaningful share of errors, as in a real audit where people can check
only a few hundred items. A v2 would give every detector a fixed flagging budget (for example its top 5% most
suspicious items) and/or use selective modern judges like Gemma-4. Ranking judges by suspicion needs a score,
not a yes/no verdict, so v2 may need the judges' token probabilities, which this run did not store. v2 must be
pre-registered as a new experiment; nothing in this run may be re-scored to rescue v1.
