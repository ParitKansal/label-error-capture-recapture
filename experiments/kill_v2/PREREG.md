# Pre-registration: kill experiment v2

**Everything here is fixed before any GPU run and before any detector output is joined to the MMLU-Redux
expert labels.** The commit hash of this file goes into `RESULTS.md`. Anything decided after results are
seen is labelled exploratory.

**v2 is a new experiment, not a re-scoring of v1.** Nothing in `experiments/kill_v1/` is re-analysed, and no
v1 output may be used to rescue a v1 conclusion. v1 is closed at NO-GO.

Written 2026-10-07.

---

## 1. Why there is a v2

v1 answered its question and the answer was NO-GO, for a reason that was about the detectors and not about
the statistics (`../kill_v1/RESULTS.md`). The estimator worked: it put 181.9 against an expert count of 181
and its interval covered the truth in 200 of 200 injection replicates. But the four 2024-era judges each
said "this key is wrong" about half the time, so together they flagged **4,145 of 5,700 items (73%)** and
caught 171 of 181 key errors. The union already found **94.5%** of the errors, so there was nothing left to
estimate and conditions (c) and (d) failed.

The fix is not a better estimator. It is to make the detectors behave like a real audit, where a person can
check a few hundred items and no more.

**v2's one substantive change: every detector flags a fixed budget of its most suspicious items, ranked by a
score.** That needs a score rather than a yes/no verdict, which v1 did not store.

The motivating observation, exploratory in v1 and not re-used as evidence here:
`google/gemma-4-31B-it-qat-w4a16-ct` flagged 730 items, caught 108 of 181 key errors — **recall 59.7%,
precision 14.9%**. A selective modern model misses about 40% of the errors, which is exactly the regime
where estimating the missed ones is worth doing.

## 2. Truth, injection, estimators, GO: unchanged from v1

Identical to `../kill_v1/PREREG.md` §2, §4, §5, §7, §8, and not restated here except where v2 differs:

- **Primary truth**: the 181 MMLU-Redux answer-key errors at pinned revision
  `372ea425445d51e1ba1188c56e5e893f8138621f`. The 32 `expert` items are excluded from the primary truth and
  reported separately.
- **A capture is a verified flag**: a flag on a non-error is a false positive and is not a capture.
- **Injection arm**: ≥ 200 replicates, 266 items (STEM 86, humanities 60, social sciences 57, other 63),
  2 of 3 wrong keys scored per `ok` item, the same per-item option shuffle, seeds `20261006 + 0 … 199`.
- **Primary estimator**: log-linear MSE, forward AIC over two-way interactions, stability guard at 10x,
  goodness of fit reported before the interval is believed. **The interval is a NONPARAMETRIC bootstrap,
  B = 2,000, seed `20261006` — a change from v1, decided from simulation alone and explained in §5a.**
- **Secondary**: Mh jackknife, Chao lower bound, Chapman per pair, union.
- **GO** needs all of (a) coverage ≥ 85% over the replicates, (b) the pooled Redux interval covers 181,
  (c) the union misses ≥ 20% of the 181, (d) |estimate − truth| ≤ ½|union − truth| on both arms.
  **NO-GO** if the union finds ≥ 90% of the Redux key errors, or (a) fails, or (b) and (d) both fail.

Condition (c) is the one v1 failed, and it is the one the budget rule is designed to make meaningful. It is
**not** relaxed: if budgeted detectors still find ≥ 90% of the errors, that is a NO-GO and v2 is closed too.

## 3. Detectors

Four, from three model families plus one statistical method. All verified on the Hugging Face Hub and
against vLLM's registry and supported-models list on 2026-10-07.

| ID | Model | Family | Size on disk | Licence | Gated | vLLM architecture |
|---|---|---|---|---|---|---|
| D1 | `google/gemma-4-31B-it` | Gemma 4 | 62.5 GB | apache-2.0 | no | `Gemma4ForConditionalGeneration` |
| D2 | `Qwen/Qwen3.8-27B-FP8` | Qwen 3.8 | 30.9 GB | apache-2.0 | no | `Qwen3_5ForConditionalGeneration` |
| D3 | `mistralai/Mistral-Small-4-119B-2603-NVFP4` | Mistral Small 4 | 70.8 GB | apache-2.0 | no | `MistralLarge3ForCausalLM` (see below) |
| D4 | confident learning over D1–D3's averaged option probabilities | — | — | — | — | — |

**Fallback, pre-registered:** if D1 does not fit alongside its KV cache, use
`google/gemma-4-31B-it-qat-w4a16-ct` (23.3 GB, same architecture, same licence). No other substitution is
permitted.

**How D3 is resolved by vLLM.** The repo has no `config.json`; it ships Mistral's native format
(`params.json`, `tekken.json`, `consolidated.safetensors.index.json`). vLLM detects a Mistral repo and uses
its mistral config path, where `adapt_config_dict` maps a MoE config with `num_shared_experts > 0` plus a
`llama_4_scaling` block to `MistralLarge3ForCausalLM`. `params.json` has `num_shared_experts: 1` and a
`llama_4_scaling` block, and `MistralLarge3ForCausalLM` is in the v0.31.0 registry and supported-models
list. NVFP4 is native on Blackwell.

**Thinking is off for every judge.** v1 lost its Qwen3.8 arm because the model spent all 24 output tokens
reasoning and never reached a verdict (`../kill_v1/RESULTS.md` §9b, 9.6% unparsed).

| Model | Switch | Why it is enough |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | `enable_thinking=False` | its chat template then prefills `<think>\n\n</think>\n\n`, so generation starts at the answer |
| `mistralai/Mistral-Small-4-...` | `reasoning_effort="none"` | the template accepts only `'none'` or `'high'` and already defaults to `'none'`; passed explicitly so it cannot drift |
| `google/gemma-4-31B-it` | none needed | no thinking switch in its template |

**Before the real pass, each notebook runs `test_first_token_is_verdict` on 8 items and stops if the first
generated token is not `CORRECT` or `INCORRECT`.** A thinking model that slips through is a failed run, not
a detector that flags nothing.

**Optional D5 is dropped.** The candidate was a 4-bit Llama-4-Scout-17B-16E-Instruct community quant
(`RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16` is the only vLLM-shaped one: ungated,
compressed-tensors W4A16). It was to be included only if a quick test showed it loads and returns verdict
logprobs. **That test needs a GPU, which this session does not have, so D5 is dropped rather than added
untested.** Four detectors is the pre-registered design. Recorded so a later version can pick it up.

## 4. The flag rule: a score and a fixed budget

**This is the change that defines v2.**

**Primary, and the only rule any GO condition depends on.** Each judge scores every item with
**P(INCORRECT)**, taken from the logprobs of the first generated verdict token and renormalised over the two
verdict words `CORRECT` and `INCORRECT`:

```
P(INCORRECT) = exp(l_INCORRECT) / (exp(l_INCORRECT) + exp(l_CORRECT))
```

Tokens are matched case-insensitively after stripping whitespace, and several spellings of the same verdict
are pooled. An item whose top-k logprobs contain neither verdict word is **unscored**; unscored items sort
last and are never flagged, which is the conservative direction. The unscored rate is reported per model.

**Each judge flags exactly its top 285 items (5% of 5,700) by P(INCORRECT).** Ties break by item index
ascending, so the flag set is fully determined by the scores.

**D4 flags the 285 items with the lowest self-confidence**, where self-confidence is the averaged D1–D3
probability assigned to the claimed key. Low means suspicious.

The budget is 285 because 5% of a 5,700-item benchmark is about what a careful human audit can re-check,
and because v1's failure was precisely that its detectors flagged 73% of the benchmark. **285 is fixed here,
before any score exists.** It is not tuned, and it is not revisited after seeing results.

**Secondary, reported only, never used for any GO condition:** the natural `INCORRECT` verdict, parsed from
the `VERDICT:` line. Reported so v2 can be compared with v1 like for like. v2's parser accepts **only** the
`VERDICT:` line; v1's bare-word fallback, which could read "correct" out of reasoning prose, is removed.

**Why this does not smuggle in tuning.** The budget fixes how *many* items each detector flags, not *which*
ones, and it is identical for every detector and every arm, including the injection replicates. No threshold
is chosen by looking at expert labels, and no score is calibrated against truth.

## 5. Contamination

All three judges postdate MMLU-Redux (June 2024), so **the Redux arm cannot be read as a clean
memorisation-free measurement** and the injection arm is the main test. This is a real weakening relative to
v1, where the judges predated the benchmark, and it is stated here rather than discovered later.

What follows from it:
- A **low** miss rate on Redux is uninformative: contamination can only make these models look better.
- A **high** miss rate on Redux is conservative and therefore meaningful.
- The injected errors are new and cannot have been memorised, so condition (a), the coverage condition, is
  unaffected.
- Option shuffling (the same per-item permutation as v1) still removes position-coded recall but not
  content memorisation.

## 5a. The interval is nonparametric in v2, and v1's was parametric

**This is a v2 change from v1, decided from the simulation in §6 alone, before any v2 detector was run and
with no expert label involved.** It is recorded here because reversing a v1 choice needs a reason on the
record.

v1 pre-registered the **parametric** bootstrap: five detectors, one of them (confident learning over four
*other* models' probabilities) close enough to independent, and under that structure simulating a whole
population from the fitted model and keeping the detected part reached nominal coverage where resampling
did not.

v2's structure is different in two ways that matter: **four detectors instead of five**, and **D4 built from
D1–D3's own probabilities**, so it loads on the same latent factor as the judges rather than cutting across
them. Under that structure the parametric interval undercovers, and it undercovers worst in exactly the
scenario the GO rule depends on:

| scenario | parametric | nonparametric |
|---|---|---|
| `d4-dep` — all four loaded, the honest model of v2's design | **74%** | 92% |
| `lowrec-d4dep` — low recall under the budget, all four loaded | **82%** | 92% |
| `d4-dep-n266` — **the injection-sized population GO (a) is measured on** | **84%** | 88% |

Across all nine scenarios the nonparametric interval is at or above the parametric one in **seven**, and it
holds **88–95% in every scenario without an invisible subpopulation**. The parametric interval is better
only in the two easiest rows (`budget-indep` 100% against 95%, `budget-dep` 95% against 94%), which are not
the regime this experiment runs in.

**The parametric interval stays in the output as a reported secondary**, so v1 and v2 can be compared like
for like. No threshold in §2 changes: GO (a) is still 85%.

## 6. Simulation evidence

Recorded before any GPU run. The v2 scenarios model the budget regime: **four** detectors rather than five,
per-detector recall around 0.25–0.45 rather than 0.7–0.9, and D4 loading on the same latent factor as
D1–D3, because D4 is built from those same models' probabilities and is not independent of them.

9 scenarios, 200 replicates each, 200 bootstrap draws, 6 worker processes with one BLAS thread each, rows
flushed as they land. Monte Carlo standard error on a coverage near 95% is 1.5 percentage points, so treat
differences under about 3 points as noise. Raw output: `simulation_coverage.txt`.

Each cell is coverage of the 95% interval / median estimate divided by truth. `miss` is the share of true
errors no detector found. **`loglin-npar` is the v2 primary** (§5a); `loglin-par` is the reported secondary
and was v1's primary. Bold marks coverage below the 85% threshold in GO (a).

| scenario | truth | miss | **loglin-npar** (primary) | loglin-par (secondary) | jackknife | chao | union |
|---|---|---|---|---|---|---|---|
| budget-indep (4 budgeted, independent) | 181 | 19% | 95% / 0.99 | 100% / 0.99 | 8% / 1.22 | 84% / 1.07 | 0% / 0.81 |
| budget-dep (3 judges share a factor) | 181 | 25% | 94% / 0.96 | 95% / 0.96 | 86% / 1.03 | 80% / 0.91 | 0% / 0.75 |
| d4-dep (all 4 loaded, rho 1.5) | 181 | 29% | 92% / 0.92 | **74%** / 0.92 | 78% / 0.95 | 56% / 0.85 | 0% / 0.71 |
| d4-dep-strong (all 4 loaded, rho 2.5) | 181 | 34% | 92% / 0.91 | 90% / 0.91 | **18%** / 0.82 | **19%** / 0.76 | 0% / 0.66 |
| lowrec-d4dep (low recall, all 4 loaded) | 181 | 39% | 92% / 0.87 | **82%** / 0.87 | **21%** / 0.84 | **26%** / 0.75 | 0% / 0.61 |
| invis-15 (15% undetectable) | 181 | 36% | **80%** / 0.81 | **71%** / 0.81 | **42%** / 0.86 | **24%** / 0.78 | 0% / 0.64 |
| invis-30 (30% undetectable) | 181 | 47% | **56%** / 0.66 | **40%** / 0.66 | **0%** / 0.71 | **4%** / 0.64 | 0% / 0.53 |
| worst (low recall, all 4 loaded, 20% invisible) | 181 | 51% | **78%** / 0.69 | **62%** / 0.69 | **0%** / 0.66 | **5%** / 0.60 | 0% / 0.49 |
| d4-dep-n266 (injection-sized) | 266 | 29% | 88% / 0.92 | **84%** / 0.92 | 75% / 0.95 | **36%** / 0.85 | 0% / 0.71 |

What this establishes:

- **The budget rule puts the experiment in the regime v1 could not reach.** The union misses 19–51% of true
  errors here, against 5.5% in the v1 run. Condition (c) needs ≥ 20%, so v2 is at least testable.
- **The primary interval holds 88–95% wherever errors are reachable at all**, including the
  injection-sized scenario GO (a) is measured on (88%) and the honest all-four-dependent model (92%).
- **Dependence is survivable; invisibility is not**, again. 15% invisible drops the primary to 80% and 30%
  drops it to 56%, while making D4 fully dependent costs nothing at all.
- **Both secondaries collapse under dependence.** The jackknife falls to 18–21% and Chao to 19–26% once all
  four detectors load on the same factor. Neither can rescue a failed primary, and neither may be swapped
  in after a bad result.
- **The estimator underestimates when it breaks** — the primary's median ratio never exceeds 0.99 and falls
  to 0.66. A GO is conservative.


## 7. Compute

Target **≤ 6 GPU-hours, hard stop 7**. All notebooks default to the **G4 runtime** (Blackwell, ~96 GB);
the pre-registered fallback runtime is H100, and on H100 the D3 notebook may need the smaller D1 fallback
model instead.

Prompt counts are unchanged from v1 at 16,360 per judge (5,700 Redux + 10,660 injection scoring), 49,080
across the three judges. The scoring pass now needs `logprobs` on the first generated token; this does not
change the prompt count.

| Step | If it works | Worst case |
|---|---|---|
| Smoke test plus the first-token-is-verdict check, every notebook | 0.3 | 0.6 |
| D1 `gemma-4-31B-it`, 16,360 prompts and the option-probability pass | 1.2 | 1.9 |
| D2 `Qwen3.8-27B-FP8`, same | 1.0 | 1.6 |
| D3 `Mistral-Small-4-119B-NVFP4`, same (119B A6B MoE, the largest download) | 1.6 | 2.4 |
| Analysis | CPU, 0 | 0 |
| **Total** | **4.1** | **6.5** |

If the running total threatens 7, cut in this order: the Redux-arm secondary verdict reporting needs no
extra GPU time, so the only cuttable item is **D3**, and dropping it leaves three detectors (D1, D2, D4),
which the simulation covers. Stop and report if the run passes 7 GPU-hours.

## 8. Known limitations

1. **All three judges postdate MMLU-Redux** (§5). The Redux arm is contaminated by construction; the
   injection arm carries the weight.
2. **D4 is not independent of D1–D3.** It is built from their probabilities. §6 models this directly.
3. **Four detectors, not five.** D5 was dropped untested (§3).
4. **Invisible errors remain untestable on real data.** v1's simulation showed 15% invisible drops coverage
   to 53% and 30% drops it to 10%; v2 re-measures this under the budget regime in §6. A NO-GO still cannot
   distinguish "the method does not work" from "this benchmark has errors nothing can see".
5. **The 5% budget is a choice, not a derivation.** A different budget would give different recall. It is
   fixed in advance and reported as the design it is.

## 9. Files and seeds

| | |
|---|---|
| Dataset revision | `372ea425445d51e1ba1188c56e5e893f8138621f` |
| Master seed | `20261006` (same as v1, so the shuffle and the 2-of-3 key choice match) |
| Injection replicate seeds | `20261006 + 0 … 199` |
| Bootstrap | B = 2,000, seed `20261006`, **nonparametric** (§5a); parametric reported as secondary |
| Flag budget | 285 items per detector |
| Drive folder | `MyDrive/label_error_crc/kill_v2` (v1's folder is never written to) |
| Estimators | `src/crc.py` |
| Detector logic | `src/detectors.py` |
| Tests | `src/test_crc.py`, `src/test_detectors.py` |
