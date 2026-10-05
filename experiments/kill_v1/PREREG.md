# Pre-registration: kill experiment v1

**Status: work in progress, not yet ready for approval.** Section 6 is provisional (the full simulation
study has not completed) and the 2-of-3-keys change the author asked for is not yet applied; both are the
next steps in `STATUS.md`. Sections 1-5 and 7-13 are the intended final text.

**Everything in this file is fixed before any GPU run and before any detector output is joined to the
MMLU-Redux expert labels.** The commit hash of this file goes into `RESULTS.md`. Anything decided after
results are seen is labelled exploratory and reported separately, never as a confirmatory result.

Written 2026-10-06, after gate 1. Design decisions taken by the author at gate 1 are marked **[gate 1]**.

---

## 1. Question and claim

Can capture–recapture over several imperfect label-error detectors estimate the **total** number of label
errors in a benchmark — including errors no detector found — with a 95% interval that actually covers the
truth?

The claim under test is not that the statistics are new. Log-linear multiple-systems estimation is from
Fienberg (1972) and software inspection used it on defect counts in the 1990s (Briand et al. 2000). The claim
is that the transfer works on benchmark label errors, where the detectors are language models that may all
fail on the same items.

## 2. What counts as truth

**Primary truth — MMLU-Redux answer-key errors.** An item is a true error if its `error_type` is
`wrong_groundtruth`, `no_correct_answer` or `multiple_correct_answers`. On the pinned revision
`372ea425445d51e1ba1188c56e5e893f8138621f` that is **181 of 5,700 items (3.18%)**. Counts per supercategory
and all other dataset facts are in `DATA.md`.

**The 32 `expert` items are excluded from the primary truth.** **[gate 1]** They are items the MMLU-Redux
annotators could not resolve, so they are neither errors nor `ok`. They stay in the benchmark (detectors see
them), they are reported separately, and they are never silently counted as `ok`. Flags on them are counted
in neither the numerator nor the denominator of precision.

**Secondary truth — any `error_type` other than `ok`** (370 items, 6.49%). Reported, not part of the GO rule.

**MMLU-Redux is itself one imperfect annotation pass.** **[gate 1]** It is a single human re-annotation with
a search-the-web protocol. Some items labelled `ok` are probably errors that the annotators missed. So the
Redux arm of this experiment is a **realism test, not an exact-truth test**: it asks whether the method
behaves sensibly on real detector outputs and real error structure. Detector flags on `ok` items are reported
as **exploratory** — they may be false positives or they may be errors Redux missed, and this experiment
cannot tell the two apart.

**The injected errors are the main calibration test.** **[gate 1]** There the truth is known exactly by
construction, so coverage measured there is a real coverage measurement. The GO rule is built on it.

## 3. Detectors (capture occasions)

Five detectors **[gate 1]**, from four model families plus one statistical method, so that no single family's
blind spot can dominate the overlap structure.

| ID | Detector | Family | Hugging Face id |
|---|---|---|---|
| D1 | LLM judge | Meta Llama 3 | `meta-llama/Meta-Llama-3-8B-Instruct` |
| D2 | LLM judge | Mistral | `mistralai/Mistral-7B-Instruct-v0.3` |
| D3 | LLM judge | Qwen 2 | `Qwen/Qwen2-7B-Instruct` |
| D4 | LLM judge | Microsoft Phi-3 | `microsoft/Phi-3-medium-4k-instruct` |
| D5 | Confident learning | — (statistical) | `cleanlab.filter.find_label_issues` |

All four judge models were uploaded to the Hub before MMLU-Redux was released; see §9.

**Judge protocol.** One prompt per (item, claimed key). The prompt shows the question, the four options and
the claimed correct answer, and asks whether the claimed answer is correct. The output format is
**verdict first** **[gate 1]**, to keep generation short:

```
VERDICT: CORRECT|INCORRECT
REASON: <at most one short sentence>
```

Generation is capped at 24 tokens, temperature 0, the same prompt template for every model, with each
model's own chat template applied. **The flag rule is fixed now: D flags the item if the parsed verdict is
`INCORRECT`.** There is no threshold to tune and no score to calibrate. A response that does not parse is
recorded as `unparsed` and counted as **not flagged**; the unparsed rate is reported per model, and a model
whose unparsed rate exceeds 5% is reported as broken rather than quietly treated as a detector that flags
nothing.

**Confident learning (D5).** Each of the four models scores all four options for every item *without seeing
the key*, giving a probability over the four options. The four models' probabilities are averaged, and
`cleanlab.filter.find_label_issues` is run with `filter_by="prune_by_noise_rate"` and default parameters
against the claimed key. D5 flags the items cleanlab returns. No parameter is tuned.

Two consequences worth stating in advance, because they shape the compute plan:

- The option probabilities **do not depend on the claimed key**, so one pass over the 5,700 items serves the
  Redux analysis *and* every injection replicate. D5 costs nothing extra per replicate.
- The probability pass runs inside the same notebook as that model's judge pass, while the weights are
  already loaded, so D5 needs no separate GPU session.

**Dependence is the expected failure mode.** D1–D4 are all instruction-tuned transformers and may miss the
same subtle errors. That is why D5 comes from a different method entirely, and why the primary estimator
carries two-way interaction terms (§5).

## 4. What counts as a capture

**An item is captured by a detector only if the detector flags it AND it is a true error.** A flag on a
non-error is a false positive: it is recorded, it is counted in the audit-cost numbers, and it is **not** a
capture. This is the single most important rule in this file. It mirrors the real audit protocol — a human
checks only the union of flagged items — and it is the direct lesson from the previous project, where false
positives on easy items decided the outcome.

Per detector we also report precision, recall and the number of flags, on both arms.

## 5. Estimators

Implemented in `src/crc.py`; tested in `src/test_crc.py`. Only numpy and scipy are required.

**Primary (the one the GO rule depends on): log-linear multiple-systems estimation.** Fienberg (1972).
A Poisson log-linear model is fitted to the 2^5 − 1 = 31 observable capture patterns; the estimate is the
found count plus the fitted count of the unobserved all-zero cell. Terms: all five main effects, plus two-way
detector interactions added by **forward selection on AIC** (deterministic, no human choice). Three-way and
higher interactions are not considered.

- **Confidence interval: a parametric bootstrap, B = 2,000, seed 20261006.** A whole population is simulated
  from the fitted model and only the detected part is kept, so the interval carries the uncertainty in the
  part nobody detected. Model selection is re-run inside every replicate, so the interval also pays for
  selection uncertainty. This choice was made from the simulation study in §6 alone, before any detector was
  run and with no expert labels involved; the alternative (resampling the found errors) reaches only 82–93%
  coverage against the parametric bootstrap's 90–100%.
- **Stability guard, fixed now:** if the selected model implies more than **10 ×** the found count, or the
  fit fails to converge, the fit is declared unstable, the independence model is used instead, and the
  estimate is reported as unstable. The 10 × figure comes from simulation only.
- **Goodness of fit is reported before the interval is believed:** deviance, degrees of freedom and the
  chi-square p-value for the selected model. A poor fit is reported as a poor fit. This is the second lesson
  from the previous project.

**Secondary estimators** (reported in full, never used to decide GO):

| Estimator | Source | Role |
|---|---|---|
| Mh jackknife | Burnham & Overton (1978, 1979), order by their sequential test | heterogeneous detectability **[gate 1]** |
| Chao lower bound | Chao (1984), bias-corrected form, log-normal interval | a floor, not a point estimate |
| Chapman | cited through Rivest & Yauck (arXiv 2307.11705) | every one of the 10 detector pairs |
| Union | — | the baseline this project is trying to beat |

The jackknife coefficients in `crc.py` are **not transcribed** from Burnham & Overton. They are re-derived by
Richardson extrapolation, and `test_crc.py` checks the derived estimator against the published closed forms
for orders 1, 2 and 3 in exact arithmetic over 300 cases.

**Dropped from the plan: the Chao & Tsay (1998) sample-coverage estimator.** It is closed access and we have
not read it. Implementing an estimator from a paper we have not read would be worse than leaving it out.

## 6. Simulation evidence, recorded before any GPU run

> **PROVISIONAL — this section is not final and this file is not ready for approval.**
> The numbers below come from the small configuration (`--quick`: 60 replicates per scenario, 100 bootstrap
> draws), which completed. The full study (400 replicates, 300 draws) was started and did **not** finish, so
> no full-size numbers exist yet. At 60 replicates the Monte Carlo standard error on a coverage near 95% is
> about 2.8 percentage points, so treat every coverage figure here as +/- 6 points and do not read small
> differences between scenarios. Section 8's GO threshold must not be judged against this table.

`python src/test_crc.py` runs seven unit tests (all passing) and the coverage study below. Scenarios cover
independent detectors, detector-specific rates, heterogeneous item detectability, and the project's own
failure mode: four judges sharing a latent factor while confident learning stays conditionally independent.

Each cell is coverage of the 95% interval / median estimate divided by truth. `miss` is the share of true
errors no detector found. `loglin-par` is the pre-registered primary; `loglin-npar` is the same point
estimate with a resampling interval, shown only to justify the choice between them.

| scenario | truth | miss | loglin-par | loglin-npar | jackknife | chao | union |
|---|---|---|---|---|---|---|---|
| M0 (independent, equal rates) | 181 | 16% | **98% / 1.00** | 90% / 1.00 | 8% / 1.19 | 87% / 1.04 | 0% / 0.83 |
| Mt (independent, per-detector rates) | 181 | 11% | **100% / 1.00** | 90% / 1.00 | 25% / 1.15 | 85% / 1.03 | 0% / 0.90 |
| Mh (item detectability varies, sd 1.0) | 181 | 18% | **95% / 0.95** | 93% / 0.95 | 57% / 1.07 | 93% / 0.97 | 0% / 0.82 |
| Mh-strong (sd 1.8) | 181 | 25% | **90% / 0.93** | 87% / 0.93 | 85% / 0.93 | 55% / 0.87 | 0% / 0.75 |
| dep-judges (4 judges share a factor, rho 1.5) | 181 | 18% | **98% / 0.97** | 90% / 0.97 | 63% / 1.07 | 93% / 0.97 | 0% / 0.83 |
| dep-strong (rho 2.5, designed to fail) | 181 | 20% | **98% / 0.97** | 98% / 0.97 | 70% / 1.11 | 95% / 0.98 | 0% / 0.81 |
| dep-4det (only 4 detectors) | 181 | 19% | **97% / 0.98** | 82% / 0.98 | 52% / 1.11 | 88% / 1.00 | 0% / 0.81 |
| dep-n266 (injection-sized population) | 266 | 17% | **98% / 0.98** | 90% / 0.98 | 43% / 1.10 | 93% / 0.98 | 0% / 0.83 |

Readings, all provisional until the full study is rerun:

- The primary estimator with the parametric bootstrap sits at or near nominal coverage in every scenario,
  including the designed-to-fail case where the four judges are strongly dependent. That is the result this
  design needs, and it is the reason the parametric bootstrap is the pre-registered interval: the
  resampling interval is visibly worse in the same runs.
- The **Mh jackknife runs high and its interval is too narrow** except under strong heterogeneity. It is
  reported and it is not what GO depends on. Recorded now so it is not rediscovered after results.
- The **four-detector scenario is worse than the five-detector one** on the resampling interval, which is
  the direction Briand et al. (2000) predict: too few capture occasions and no model is reliable. This is
  the evidence for five detectors rather than four. The difference is within Monte Carlo error at this
  replicate count and needs the full rerun to stand up.
- Chao sits below the truth under heterogeneity, as a floor should, and runs slightly high under
  near-homogeneous capture. It is read as a floor only.

## 7. The injection experiment (the main calibration test)

**Scoring, on GPU, once.** Each judge scores **every `ok` item against each of its 3 wrong keys**
**[gate 1]**: 5,330 × 3 = 15,990 extra prompts per judge. Together with the 5,700-item Redux pass that is
**21,690 prompts per judge**. Nothing about the injection is decided on GPU; the GPU only produces, for every
(`ok` item, candidate wrong key) pair, that judge's verdict.

**Replicates, on CPU, at analysis time.** **≥ 200 injection replicates** **[gate 1]**, seeds
`20261006 + 0 … 199`, fixed here. Each replicate:

1. Draw an injection set of **266 items** (5% of the 5,330 `ok` items), stratified by MMLU supercategory in
   proportion to each stratum's `ok` count: **STEM 86, humanities 60, social sciences 57, other 63**
   (largest-remainder rounding of 5%, fixed here so it cannot drift).
2. For each drawn item, flip the key to one of its 3 wrong options, drawn uniformly at random.
3. Build the benchmark for that replicate: drawn items carry the flipped key, every other item carries its
   original key.
4. Read off each judge's verdict for every item from the scored table — flipped items from the injection
   scoring, the rest from the Redux pass. Recompute D5 from the stored option probabilities against that
   replicate's keys.
5. Truth for that replicate is the 266 injected errors, known exactly. Captures are verified flags on those
   266 items. Run every estimator and record whether the primary 95% interval covers 266.

**Coverage is then the share of replicates whose interval covers 266** — an empirical coverage measurement,
not an asymptotic claim.

Reported alongside: the mean union count over replicates, the mean |estimate − 266|, the mean |union − 266|,
the distribution of selected interaction terms, the share of replicates flagged unstable, and the
goodness-of-fit p-value distribution.

## 8. GO / NO-GO

**[gate 1]** GO requires **all four**:

| | Condition |
|---|---|
| (a) | Empirical coverage of the primary 95% interval over the ≥ 200 injection replicates is **≥ 85%** |
| (b) | The pooled primary 95% interval on MMLU-Redux **covers 181** |
| (c) | The union of all five detectors **misses ≥ 20%** of the 181 Redux key errors (otherwise there is no problem to solve) |
| (d) | **\|estimate − truth\| ≤ ½ \|union − truth\|**, both on Redux and on average over the injection replicates |

**The primary target is the pooled estimate over all 5,700 items. The four supercategories are secondary:
reported, not part of GO.** **[gate 1]** This replaces the "3 of 4 supercategories" rule in `PLAN.md`, which
would have been decided by the social-sciences stratum and its 16 true errors.

**NO-GO** if the union already finds ≥ 90% of the Redux key errors, or if (a) fails, or if (b) and (d) both
fail. A single failure among (b), (c), (d) with (a) passing is reported as a **partial** result with the
failing condition named — not quietly upgraded to GO.

**What is reported either way**, written down now so neither outcome can be reshaped later:

- every detector's flags, precision and recall, on both arms;
- the full capture-pattern table (all 31 cells) for Redux and the replicate average;
- every estimator's point estimate and interval, with goodness of fit;
- the estimated number of errors **no detector found**, with its interval — the number this project exists
  to produce;
- the audit-cost reading: how many items a human would have to check to find what the union found, and what
  the estimate says is still out there;
- on NO-GO, which of (a)–(d) failed and the most likely reason, written up as a public negative result in the
  style of the previous project.

## 9. Contamination control

MMLU-Redux first appeared on arXiv on **2024-06-06**. A model whose weights were uploaded before that date
cannot have memorised the corrections.

| Model | HF repo created | Stated cutoff | Margin |
|---|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | 2024-04-17 | March 2023 | clear |
| `mistralai/Mistral-7B-Instruct-v0.3` | 2024-05-22 | not stated on the card | clear by upload date |
| `Qwen/Qwen2-7B-Instruct` | 2024-06-04 | not stated on the card | **2 days** |
| `microsoft/Phi-3-medium-4k-instruct` | 2024-05-07 | October 2023 | clear |

If a stricter margin is wanted later, the **pre-registered** substitute for D3 is `Qwen/Qwen1.5-7B-Chat`.
Swapping it in for any other reason is not permitted.

- The injected errors are new, so no model can have memorised them. The flip lists are generated from the
  seeds in §7 at analysis time.
- **Exploratory** check: repeat the judges with `Qwen/Qwen2.5-7B-Instruct` (uploaded 2024-09-16) and
  `meta-llama/Llama-3.1-8B-Instruct` (2024-07-18) on the Redux pass and a fixed 1,500-item injection
  subsample. If the newer models flag the Redux errors much better than the old ones but are no better on
  the injected errors, that is contamination, and it is reported as such.
- Expert labels are joined to detector outputs only in the analysis notebook, after this file is committed.

## 10. Compute

Measured against the hard stop of 7 A100-hours in `PLAN.md`. Prompt counts: 21,690 per judge
(5,700 Redux + 15,990 injection scoring), 86,760 across the four judges. D5 adds no session, because the
option-probability pass runs in the judge notebook while the weights are loaded.

| Step | If it works | Worst case |
|---|---|---|
| Smoke test: every notebook on 50 items | 0.4 | 0.8 |
| 4 judge notebooks in parallel, 21,690 prompts each, plus option probabilities for D5 | 2.5 | 4.0 |
| Contamination check, 2 newer models (Redux pass + 1,500-item injection subsample) | 0.5 | 1.0 |
| 6th detector from another family, only if dependence is extreme | 0 | 0.6 |
| Analysis, all 200+ replicates and all bootstraps | CPU High-RAM, 0 | 0 |
| **Total** | **3.4** | **6.4** |

**Pre-registered trim rule.** Each notebook reports measured prompts per second at the end of the smoke test.
If the projected total for the four judge notebooks exceeds **4.5 A100-hours**, the injection scoring pool is
cut from all 5,330 `ok` items to a **stratified random 2,400** (seed 20261006), giving 12,900 prompts per
judge instead of 21,690 — a 41% cut. Injection replicates are then drawn from that pool. This rule triggers
on measured throughput only; it has nothing to do with any label, and it is fixed here so it cannot be
invoked selectively later.

Stop and report if the experiment passes 7 A100-hours, whatever state it is in.

## 11. Exploratory, not confirmatory

Labelled as exploratory wherever they appear, and never used to support the main claim:

- detector flags on items MMLU-Redux labelled `ok` (§2);
- anything about the 32 `expert` items;
- the newer-model contamination comparison (§9);
- the four supercategory estimates (§8);
- the secondary truth (any non-`ok` error type);
- the "how many detectors are enough" curve obtained by re-running the estimators on detector subsets;
- any estimator, threshold or stratification not named in this file.

## 12. What would invalidate the run

Stated now so it cannot be argued about later. Any of these means the run is reported as failed rather than
reinterpreted: a judge's unparsed rate above 5%; a detector flagging more than 50% or fewer than 0.5% of
items (it is not a detector); the analysis notebook reading expert labels before this file's commit hash is
recorded; the pinned dataset revision not matching `DATA.md`; or the GPU budget passing 7 A100-hours.

## 13. Files and seeds

| | |
|---|---|
| Dataset revision | `372ea425445d51e1ba1188c56e5e893f8138621f` |
| Master seed | `20261006` |
| Injection replicate seeds | `20261006 + 0 … 199` (≥ 200 replicates) |
| Bootstrap | B = 2,000, seed `20261006` |
| Estimators | `src/crc.py` |
| Tests and simulation | `src/test_crc.py` |
| Dataset facts | `DATA.md`, regenerated by `src/dataset_facts.py` |
| Results go to | `RESULTS.md`, which records this file's commit hash |
