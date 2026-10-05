# Dataset facts: MMLU-Redux 2.0 (phase 0)

All numbers below were computed on 2026-10-06 from the dataset files themselves, not from the dataset card.
Regenerate with `src/dataset_facts.py` (downloads the 57 Arrow files at the pinned revision and recounts).

## Pinned source

| Field | Value |
|---|---|
| Dataset | `edinburgh-dawg/mmlu-redux-2.0` (Hugging Face) |
| **Revision (pin this)** | `372ea425445d51e1ba1188c56e5e893f8138621f` |
| Revision date | 2025-02-25 |
| Licence | CC-BY-4.0 |
| DOI | 10.57967/hf/3469 |
| Paper | Gema et al., NAACL 2025, arXiv 2406.04127 |
| Files at that revision | 57 `.arrow` files, one per subject, 2.8 MB total |
| Columns | `question`, `choices`, `answer`, `error_type`, `source`, `correct_answer`, `potential_reason` |

Two independent reads agree: counting the 57 Arrow files at the pinned revision gives exactly the same per-subject
`error_type` frequencies as the Hugging Face datasets-server `/statistics` endpoint (0 mismatches over 57 subjects).

## Shape

- **5,700 items**, 57 subjects × exactly 100 items each, single `test` split per subject.
- Every item has exactly 4 choices and an `answer` index in 0–3. 5,693 of 5,700 have 4 distinct choice strings
  (7 items contain a repeated option).
- The MMLU-Redux paper's headline "57% of the analysed questions in the Virology subset contain errors" reproduces
  exactly: virology has 43 `ok` and 57 not-`ok`. This confirms we are reading the intended data.

## Error-type counts (the ground truth)

MMLU-Redux taxonomy, from the dataset card: Type 1 question assessment (`bad_question_clarity`, `bad_options_clarity`),
Type 2 ground-truth verification (`no_correct_answer`, `multiple_correct_answers`, `wrong_groundtruth`), plus `ok`
and `expert` (annotator was unsure and said the item needs more expertise).

| `error_type` | count | % of 5,700 |
|---|---|---|
| `ok` | 5,330 | 93.51 |
| `wrong_groundtruth` | 106 | 1.86 |
| `no_correct_answer` | 36 | 0.63 |
| `multiple_correct_answers` | 39 | 0.68 |
| `bad_question_clarity` | 132 | 2.32 |
| `bad_options_clarity` | 25 | 0.44 |
| `expert` | 32 | 0.56 |
| **Primary truth** (answer-key errors: `wrong_groundtruth` + `no_correct_answer` + `multiple_correct_answers`) | **181** | **3.18** |
| **Secondary truth** (any `error_type` other than `ok`) | **370** | **6.49** |

## By MMLU supercategory (the 4 strata)

Mapping taken from the official MMLU repository, `hendrycks/test` `categories.py` (opened 2026-10-06), which maps
each subject to a subcategory and each subcategory to one of 4 supercategories.

| Stratum | items | `ok` | **key errors (primary)** | rate | any error (secondary) | rate |
|---|---|---|---|---|---|---|
| STEM | 1,800 | 1,714 | **59** | 3.28% | 86 | 4.78% |
| humanities | 1,300 | 1,208 | **43** | 3.31% | 92 | 7.08% |
| social sciences | 1,200 | 1,147 | **16** | 1.33% | 53 | 4.42% |
| other (business, health, misc.) | 1,400 | 1,261 | **63** | 4.50% | 139 | 9.93% |
| **all** | **5,700** | **5,330** | **181** | **3.18%** | **370** | **6.49%** |

Breakdown within each stratum:

| Stratum | `wrong_groundtruth` | `no_correct_answer` | `multiple_correct_answers` | `bad_question_clarity` | `bad_options_clarity` | `expert` |
|---|---|---|---|---|---|---|
| STEM | 47 | 9 | 3 | 15 | 8 | 4 |
| humanities | 11 | 6 | 26 | 30 | 8 | 11 |
| social sciences | 8 | 3 | 5 | 34 | 1 | 2 |
| other | 40 | 18 | 5 | 53 | 8 | 15 |

## How the errors are distributed across subjects

Errors are extremely concentrated, which matters more for this project than the headline rate:

- **26 of 57 subjects have zero primary (answer-key) errors.**
- virology alone has 41 of the 181 key errors (23%); virology + college_chemistry = 64 (35%).
- Top subjects by key errors: virology 41, college_chemistry 23, professional_law 13, formal_logic 12,
  logical_fallacies 10, global_facts 9, abstract_algebra 8, public_relations 6.

Full per-subject table: `DATA_by_subject.md`.

## Data-quality notes that affect the design

1. **The smallest stratum has 16 true key errors.** Capture–recapture on 16 captures is at the edge of what any
   estimator can do. Briand et al. (2000) found underestimation is substantial when the number of capture occasions
   is small; small *population* size is the other half of the same problem. `PREREG.md` has to say in advance what
   happens if a stratum's estimate is unusable, and the GO/NO-GO rule "covers in at least 3 of 4 supercategories"
   needs to be read with social sciences in mind. Flagged as an open question for the gate.
2. **`correct_answer` is not a clean verification field.** It is non-empty for 101 of 106 `wrong_groundtruth` items
   and 33 of 39 `multiple_correct_answers`, but only 22 of 36 `no_correct_answer`; it is also non-empty for 286
   items labelled `ok`, where it is just an index echoing the existing key. It is sometimes an index ("3") and
   sometimes free text. Do not use it as a parse-reliable field; use `error_type` for truth.
3. **`source` is empty for 1,339 of 5,700 items** (non-empty 4,361). Not needed for the design, but it rules out
   "source found / not found" as a stratifying covariate without extra work.
4. **Duplicate questions.** 48 distinct question texts appear more than once (52 extra rows); 32 rows are exact
   duplicates of (question, choices, answer). Of the 100 rows involved, 98 are `ok`, 1 `wrong_groundtruth`,
   1 `bad_options_clarity`. Deduplicating changes the primary count by at most 1. `PREREG.md` should state the
   choice (recommendation: keep all 5,700, note the duplicates) rather than leave it open.
5. **The ground truth is itself one imperfect pass.** MMLU-Redux is a single human re-annotation with a
   search-the-web protocol, and `expert` (32 items) marks items the annotators could not resolve. So "truth" here
   means "the MMLU-Redux label", and the 32 `expert` items are neither clearly errors nor clearly `ok`.
   `PREREG.md` must fix their handling in advance (recommendation: excluded from the primary truth, reported
   separately; never silently counted as `ok`).
6. **The dataset card prose is stale in one place.** It says MMLU-Redux "comprises subsampled test splits of the
   aforementioned thirty MMLU subsets" — that is the 1.0 description. Version 2.0 at this revision has all 57.
   The counts above come from the files, so this does not affect anything.

## Injected-error set (second, fully known truth)

5% of the 5,330 `ok` items is 266.5, so a 5% flip gives **266 or 267** injected errors
(per stratum at 5%: STEM ≈ 86, humanities ≈ 60, social sciences ≈ 57, other ≈ 63).
`PREREG.md` must fix the rounding rule, the seed, and whether sampling is stratified (recommendation: stratified by
supercategory, flip to a uniformly random one of the 3 non-key options, seed written into the pre-registration).

## Detector models: contamination control

MMLU-Redux was first released on arXiv on **2024-06-06**. A model whose training data closes before that date
cannot have memorised the corrections. Upload dates are from the Hugging Face model API (`createdAt`), read
2026-10-06; cutoffs are from the model cards where the card states one.

| Model | HF repo created | Stated training cutoff | Before 2024-06-06? | Notes |
|---|---|---|---|---|
| `meta-llama/Meta-Llama-3-8B-Instruct` | 2024-04-17 | pretraining data cutoff **March 2023**; release date April 18, 2024 (official `meta-llama/llama3` `MODEL_CARD.md`) | yes, clearly | **Gated (manual approval).** The author needs an approved HF token in Colab. |
| `mistralai/Mistral-7B-Instruct-v0.3` | 2024-05-22 | **not stated on the model card** | yes, by upload date | Card is 201 lines and gives no cutoff; the upload date is the evidence. |
| `Qwen/Qwen2-7B-Instruct` | 2024-06-04 | **not stated on the model card**; Qwen2 blog dated 2024-06-07 | yes, by upload date — but only **2 days** before | Tightest case. The weights were uploaded before the MMLU-Redux arXiv post, so they cannot contain it, but there is no margin. |
| `microsoft/Phi-3-medium-4k-instruct` | 2024-05-07 | **October 2023** (card: "static model trained on an offline dataset with cutoff date October 2023"; trained Feb–Apr 2024) | yes, clearly | 14B — fits an A100 in bf16. Note the plan says "Phi-3-medium-instruct"; the actual repo is `Phi-3-medium-4k-instruct`. |

Newer models for the exploratory contamination check (both uploaded after MMLU-Redux, as intended):

| Model | HF repo created |
|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | 2024-09-16 |
| `meta-llama/Llama-3.1-8B-Instruct` | 2024-07-18 (gated) |

All four primary detector models pass the pre-registered rule, so none needs replacing. Two things to carry
forward: the Llama repos are gated, so the author must accept the licence and use a token in Colab; and the
Qwen2 margin is 2 days, so if a stricter margin is wanted, `Qwen/Qwen1.5-7B-Chat` (uploaded well before) is the
pre-registered fallback.
