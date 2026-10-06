# Pre-registration: kill experiment v1

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
model's own chat template applied.

**Options are shown in a shuffled order.** For every item, the four options are permuted by a fixed
per-item permutation and then relabelled A–D in the new order, so the claimed key moves with its option
text rather than staying at its original letter. The permutation is drawn once from
`numpy.random.default_rng([20261006, item_index])`, stored, and used **identically** by every judge, by the
confident-learning pass, by the injection scoring and by every replicate — so the detectors stay comparable
and nothing about the shuffle varies between arms.

Why: Gema et al. (§3.1) report that on Professional Law and Formal Logic, model accuracy on *erroneous*
instances was similar to or higher than on correct ones, and say this "may serve as evidence of
memorisation, suggesting that these MMLU instances were learned during the models' pretraining processes".
MMLU answer keys are position-coded A–D and are all over the public internet. A judge that has memorised
"this question's answer is C" would endorse a wrong key whenever the wrong key happens to sit at C.
Shuffling breaks position-coded recall. It does **not** break content memorisation; see §12. **The flag rule is fixed now: D flags the item if the parsed verdict is
`INCORRECT`.** There is no threshold to tune and no score to calibrate. A response that does not parse is
recorded as `unparsed` and counted as **not flagged**; the unparsed rate is reported per model, and a model
whose unparsed rate exceeds 5% is reported as broken rather than quietly treated as a detector that flags
nothing.

**Confident learning (D5).** Each of the four models scores all four options for every item *without seeing
the key*, giving a probability over the four options. The options are in the same shuffled order as the
judge prompts, and the probabilities are mapped back to the original option identities before cleanlab sees
them, so D5 is scored on option *content*, not on letters. The four models' probabilities are averaged, and
`cleanlab.filter.find_label_issues` is run with `filter_by="prune_by_noise_rate"` and default parameters
against the claimed key. D5 flags the items cleanlab returns. No parameter is tuned.

Two consequences worth stating in advance, because they shape the compute plan:

- The option probabilities **do not depend on the claimed key**, so one pass over the 5,700 items serves the
  Redux analysis *and* every injection replicate. D5 costs nothing extra per replicate.
- The probability pass runs inside the same notebook as that model's judge pass, while the weights are
  already loaded, so D5 needs no separate GPU session.

**Dependence is the expected failure mode.** D1–D4 are all instruction-tuned transformers and may miss the
same subtle errors. That is why D5 comes from a different method entirely, and why the primary estimator
carries two-way interaction terms (§5). Note that our own simulation (§6) does **not** show a coverage
penalty at four detectors; the fifth detector is here because a non-LLM detector family is worth having on
its own terms, and because Briand et al. (2000) found accuracy degrades with too few inspectors on real
data.

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
  run and with no expert labels involved: across the eight scenarios in §6 the parametric interval covers
  90–100% against the resampling interval's 90–96%, at or above it everywhere and strictly above in six.
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

`python src/test_crc.py --out sim.txt` runs seven unit tests (all passing) and the coverage study below:
**200 replicates per scenario, 200 bootstrap draws, 6 worker processes**, each scenario row flushed to disk
as it completes. Monte Carlo standard error on a coverage near 95% is **1.5 percentage points**, so treat
differences under about 3 points as noise. Raw output: `simulation_coverage.txt`.

Each cell is coverage of the 95% interval / median estimate divided by truth. `miss` is the share of true
errors no detector found. `loglin-par` is the pre-registered primary. **Bold marks coverage below the 85%
threshold in §8.**

| scenario | truth | miss | loglin-par | loglin-npar | jackknife | chao | union |
|---|---|---|---|---|---|---|---|
| M0 (independent, equal rates) | 181 | 17% | 100% / 1.00 | 96% / 1.00 | 7% / 1.19 | 90% / 1.03 | 0% / 0.83 |
| Mt (independent, per-detector rates) | 181 | 11% | 100% / 1.00 | 90% / 1.00 | 18% / 1.14 | 86% / 1.03 | 0% / 0.89 |
| Mh (detectability varies, sd 1.0) | 181 | 18% | 93% / 0.96 | 90% / 0.96 | 62% / 1.06 | 92% / 0.97 | 0% / 0.82 |
| Mh-strong (sd 1.8) | 181 | 25% | 90% / 0.94 | 90% / 0.94 | 88% / 0.95 | 60% / 0.87 | 0% / 0.75 |
| dep-judges (4 judges share a factor, rho 1.5) | 181 | 18% | 99% / 0.96 | 90% / 0.96 | 64% / 1.07 | 93% / 0.97 | 0% / 0.82 |
| dep-strong (rho 2.5) | 181 | 20% | 100% / 0.97 | 96% / 0.97 | 72% / 1.10 | 96% / 0.98 | 0% / 0.80 |
| dep-4det (only 4 detectors) | 181 | 20% | 98% / 0.98 | 92% / 0.98 | 59% / 1.11 | 94% / 0.98 | 0% / 0.80 |
| **d5-dep** (all 5 load on the factor, rho 1.5) | 181 | 24% | 89% / 0.93 | 89% / 0.93 | 84% / 0.95 | 60% / 0.88 | 0% / 0.76 |
| **d5-dep-strong** (all 5 load, rho 2.5) | 181 | 30% | 90% / 0.92 | 89% / 0.92 | 55% / 0.86 | 30% / 0.80 | 0% / 0.70 |
| **invis-15** (15% undetectable) | 181 | 30% | **53% / 0.82** | **46%** / 0.82 | 81% / 0.91 | 42% / 0.83 | 0% / 0.70 |
| **invis-30** (30% undetectable) | 181 | 42% | **10% / 0.68** | **24%** / 0.68 | 18% / 0.75 | 8% / 0.69 | 0% / 0.59 |
| **lowrec** (recall 0.17–0.28, union misses 40%) | 181 | 40% | 87% / 0.91 | 92% / 0.91 | 86% / 0.96 | 68% / 0.84 | 0% / 0.60 |
| **lowrec-d5dep** (low recall, all 5 loaded) | 181 | 43% | **78% / 0.87** | 84% / 0.87 | 39% / 0.81 | 26% / 0.73 | 0% / 0.57 |
| **worst** (low recall, all 5 loaded, 30% invisible) | 181 | 60% | **45% / 0.58** | 64% / 0.58 | 0% / 0.56 | 2% / 0.51 | 0% / 0.40 |
| dep-n266 (injection-sized population) | 266 | 18% | 99% / 0.98 | 90% / 0.98 | 55% / 1.09 | 94% / 0.98 | 0% / 0.82 |

### The failure boundary

The thresholds in §8 are unchanged. What follows is where the primary estimator stops meeting them, measured
before any GPU run, so that a failure on real data can be matched against a known cause rather than
explained after the fact.

**Primary coverage falls below 85% in four of the fifteen scenarios**, and the cause is not detector
dependence:

| What is varied | Coverage | Verdict |
|---|---|---|
| Detector dependence alone, even all 5 detectors on one strong factor (`d5-dep`, `d5-dep-strong`) | 89–90% | **holds** |
| Low recall alone, union missing 40% (`lowrec`) | 87% | **holds, with little room** |
| Low recall **and** all 5 detectors dependent (`lowrec-d5dep`) | **78%** | **fails** |
| 15% of errors undetectable (`invis-15`) | **53%** | **fails** |
| 30% of errors undetectable (`invis-30`) | **10%** | **fails badly** |
| All three together (`worst`) | **45%** | **fails** |

Three things follow, and they are the reason these scenarios were added:

1. **Dependence is survivable; invisibility is not.** Making confident learning load on the same latent
   factor as the four judges — which is the honest model of our design, since D5 is built from those same
   four models' probabilities — costs about 10 points of coverage and pulls the median estimate to 0.93 of
   truth, but coverage stays at 89–90%. By contrast, a 15% invisible subpopulation cuts coverage to 53% and
   30% cuts it to 10%. **The assumption that every error has some chance of being detected is the one that
   matters**, and no amount of estimator sophistication substitutes for it. This is the known limitation in
   §11a item 4, now quantified.
2. **Low recall and dependence interact.** Either alone is survivable (87%, 89%); together they give 78%.
   The real experiment will sit somewhere in this region, since D5 genuinely is not independent and judge
   recall on subtle key errors may well be low. **A coverage failure on the injection arm should be read
   first as this interaction, not as a bug.**
3. **Every estimator degrades the same way, so no fallback rescues it.** Under `worst` the jackknife covers
   0%, Chao 2%, the union 0%. Where the primary fails, the secondaries fail harder. There is no estimator
   in this file that can be swapped in after a bad result to recover a GO, and that is stated now so nobody
   tries.

The direction of failure is also worth recording: the primary estimator's median ratio never exceeds 1.00
in any of the fifteen scenarios and falls to 0.58 in the worst. **It underestimates when it breaks, it does
not overestimate.** A GO result is therefore conservative; a NO-GO may mean either that the method does not
work or that the benchmark contains errors nothing can see.

### Other readings

- The parametric bootstrap is the right interval but by a modest margin: at or above the resampling
  interval in eleven of fifteen scenarios and clearly better under independence (100% against 96% and 90%).
  In three of the four failing scenarios the resampling interval is *less* bad (`invis-30` 24% against 10%,
  `worst` 64% against 45%), but both are far below any usable threshold there, so this does not change the
  choice. An earlier 60-replicate run suggested a gap of 82–93% against 90–100%; the real gap is smaller
  and is corrected here.
- **The four-detector scenario shows no penalty** (98%), contrary to what the small run suggested, so our
  own simulation does not support preferring five detectors over four. The five-detector design stands on
  the gate-1 decision and on Briand et al. (2000).
- **The Mh jackknife is unusable as a point estimate**: 0–88% coverage, median up to 1.19 of truth. It runs
  high in easy scenarios and collapses in hard ones. Secondary only; GO does not depend on it.
- **Chao behaves as a floor** and degrades fast under dependence (30% coverage under `d5-dep-strong`).
- **The union never covers the truth in any of the fifteen scenarios**, missing 11–60%.

## 7. The injection experiment (the main calibration test)

**Scoring, on GPU, once.** Each judge scores **every `ok` item against 2 of its 3 wrong keys**:
5,330 × 2 = 10,660 extra prompts per judge. Together with the 5,700-item Redux pass that is **16,360 prompts
per judge**, 65,440 across the four judges. Nothing about the injection is decided on GPU; the GPU only
produces, for every (`ok` item, candidate wrong key) pair, that judge's verdict.

**Which 2 of the 3.** For each `ok` item, the two wrong options to score are drawn uniformly at random
without replacement from its three non-key options, with seed `20261006`, generated once and stored. The
selection is random per item rather than positional, so it carries no bias toward any option position. Every
judge scores the **same** two options for a given item, so the detectors stay comparable.

This replaces the earlier 3-of-3 design (15,990 extra prompts per judge, 21,690 total). The earlier design
needed a throughput-triggered trim rule to stay inside the GPU budget; at 2 of 3 the budget has room and the
trim rule is gone, so there is no conditional branch left in the compute plan.

**Replicates, on CPU, at analysis time.** **≥ 200 injection replicates** **[gate 1]**, seeds
`20261006 + 0 … 199`, fixed here. Each replicate:

1. Draw an injection set of **266 items** (5% of the 5,330 `ok` items), stratified by MMLU supercategory in
   proportion to each stratum's `ok` count: **STEM 86, humanities 60, social sciences 57, other 63**
   (largest-remainder rounding of 5%, fixed here so it cannot drift).
2. For each drawn item, flip the key to one of its **2 scored** wrong options, drawn uniformly at
   random. (Those two were themselves drawn at random from the item's three wrong options, so no
   option position is favoured.)
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

## 9. Contamination and memorisation control

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

### 9a. Exploratory: does option order matter? (600-item original-order check)

Shuffling is the pre-registered default, so the main run gives no direct evidence about position-coded
memorisation. To get that evidence, the four judges are additionally run **in the original MMLU option
order** on a fixed sample of **600 items**: **all 181 Redux key errors**, plus 419 further items drawn
stratified by MMLU supercategory from the rest, with seed `20261006`. That is 2,400 extra prompts in total
(600 x 4 judges), run inside the judge notebooks while the weights are already loaded.

Reported: for each judge, the share of Redux key errors it endorses (says `CORRECT`) in original order
against shuffled order, overall and for Professional Law and Formal Logic separately, since those are the
two subjects Gema et al. single out. A materially higher endorsement rate in original order is evidence of
position-coded memorisation.

**This is exploratory and is not part of GO.** It cannot change the primary analysis, which always uses the
shuffled order.

### 9b. Exploratory: what do modern judges still miss?

Not part of GO, and not a detector in the estimator. The question is narrow: **what share of the 181 Redux
key errors does the union of two current frontier-class open models miss?** If strong recent models still
miss a large share, the problem this project addresses is not an artefact of using small 2024 models.
Contamination can only make these models look *better*, so a large miss rate is a conservative result and a
small one proves nothing — stated now so the outcome cannot be over-read in either direction.

| Model | Verified on the Hub 2026-10-07 | Notes |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | uploaded 2026-08-13, apache-2.0, ungated, 30.9 GB of weights | **Not text-only.** The id the author gave does exist, but it is `image-text-to-text` (`Qwen3_5ForConditionalGeneration`, `language_model_only: false`, a vision tower and a video preprocessor). We use it text-only by never passing an image; the model itself is multimodal. Every Qwen 27B in this generation is multimodal, and the bf16 `Qwen/Qwen3.8-27B` is 55.6 GB, so this FP8 build is the only one that fits at all. |
| `google/gemma-4-31B-it-qat-w4a16-ct` | uploaded 2026-06-04, apache-2.0, ungated, 23.3 GB of weights | Different family, as asked. W4A16 is weight-only INT4, which runs natively on Ampere through Marlin kernels, so it is the lower-risk of the two. |

**Two risks to settle at the smoke test, not by assumption.** The Colab A100 is 40 GB and Ampere (SM80),
which has **no native FP8**: vLLM must fall back to a weight-only W8A16 path for the Qwen model, and 30.9 GB
of weights on a 40 GB card leaves under 9 GB for KV cache and activations. If the Qwen model fails to load
or runs out of memory, the **pre-registered fallback is to run this arm with the Gemma model alone and
report that**, rather than substituting a model chosen after the fact. The arm is exploratory and is first
in the cut order (§10).

Protocol: the same judge prompt, the same shuffled option order, **Redux pass only** (5,700 items each, no
injection scoring). Reported: per model and for the union of the two, recall on the 181 key errors, the
share missed, and the flag rate on `ok` items for context.
- Expert labels are joined to detector outputs only in the analysis notebook, after this file is committed.

## 10. Compute

Measured against the hard stop of 7 A100-hours in `PLAN.md`. The Colab A100 is **40 GB, Ampere (SM80)**.
Prompt counts: **16,360 per judge** (5,700 Redux + 10,660 injection scoring), **65,440** across the four
judges, plus 2,400 for the original-order check (§9a) and 11,400 for the modern arm (§9b). Option shuffling
adds **no** prompts — it changes prompt content, not prompt count. D5 adds no session, because the
option-probability pass runs in the judge notebook while the weights are loaded, and those probabilities do
not depend on the answer key, so one pass serves the Redux analysis and every injection replicate.

| Step | If it works | Worst case |
|---|---|---|
| Smoke test: every notebook on 50 items | 0.4 | 0.8 |
| 4 judge notebooks in parallel: 16,360 prompts each, option probabilities for D5, and the 2,400-prompt original-order check (§9a) | 2.2 | 3.5 |
| Contamination check, 2 newer small models (§9) | 0.5 | 1.0 |
| Modern judges arm, 2 large models in one session, Redux pass only (§9b) | 1.0 | 1.8 |
| 6th detector from another family, only if dependence is extreme | 0 | 0.6 |
| Analysis, all 200+ replicates and all bootstraps | CPU High-RAM, 0 | 0 |
| **Total** | **4.1** | **7.7** |

**Expected 4.1 fits with room. The uncut worst case, 7.7, does not.** So the cut order is pre-registered
here and triggers on the *measured running total*, not on anything a result says:

1. **Cut the modern judges arm (§9b) first** — saves up to 1.8. It is exploratory, it is not a detector,
   and it cannot change the primary analysis.
2. **Then cut the 600-item original-order check (§9a)** — saves up to 0.1.
3. Then cut the contamination check (§9) — saves up to 1.0.

With step 1 applied the worst case is **5.9**, inside the stop. The 6th-detector row is only reached if the
observed detector dependence is extreme, and if reaching it would pass 7 hours, stop and report instead.

Each notebook reports measured prompts per second at the end of its smoke test, as a check that the
projection above holds. If the four judge notebooks project past 4.5 A100-hours between them, stop and
report rather than quietly cutting scope beyond the order above.

Stop and report if the experiment passes 7 A100-hours, whatever state it is in.

## 11. Exploratory, not confirmatory

Labelled as exploratory wherever they appear, and never used to support the main claim:

- detector flags on items MMLU-Redux labelled `ok` (§2);
- anything about the 32 `expert` items;
- the newer-model contamination comparison (§9);
- the 600-item original-order check (§9a);
- the modern judges arm (§9b);
- the four supercategory estimates (§8);
- the secondary truth (any non-`ok` error type);
- the "how many detectors are enough" curve obtained by re-running the estimators on detector subsets;
- any estimator, threshold or stratification not named in this file.

## 11a. Known limitations

Stated in advance so they are not presented later as discoveries.

1. **Shuffling does not stop content memorisation.** Relabelling the options defeats position-coded recall
   ("the answer is C"), which is what Gema et al.'s Professional Law and Formal Logic finding points at.
   It does nothing about a model that has memorised the *text* of the original answer. A judge that
   remembers "the answer to this question is 'strict liability'" will still endorse that option wherever it
   now sits. The only real controls for that are the pre-June-2024 model cutoffs (§9) and the injected
   errors, which no model can have seen. Residual content memorisation stays an unquantified limitation of
   the Redux arm, and is one more reason the injection arm, not the Redux arm, carries the GO rule.
2. **MMLU-Redux is one imperfect annotation pass**, so the Redux arm is a realism test rather than an
   exact-truth test (§2).
3. **D5 is not independent of D1–D4.** Confident learning is built from the same four models' option
   probabilities, so the design has four model families, not five independent detectors. The simulation
   (§6) therefore includes scenarios where all five detectors load on the same latent factor.
4. **The invisible-error assumption is untestable on real data, and it is the assumption that matters.**
   If a share of real label errors is undetectable by every detector we have, no capture-recapture method
   can recover them, and the estimate is a floor on an unknown quantity rather than the total. §6 now
   quantifies how fast this kills the method: **15% invisible drops primary coverage to 53%, 30% drops it
   to 10%**, while even putting all five detectors on one strong shared factor still leaves 90%. Nothing in
   this design can measure that share on MMLU-Redux, so a NO-GO cannot distinguish "the method does not
   work" from "this benchmark contains errors nothing can see". Said plainly in advance.
5. **One benchmark, one language, four-option multiple choice.** Nothing here establishes that the method
   transfers to other benchmark formats.

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
