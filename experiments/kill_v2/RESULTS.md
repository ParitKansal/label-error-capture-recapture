# Kill experiment v2: results

Run on Google Colab on 2026-10-06 (judges on G4 Blackwell GPUs, analysis on a 44-core TPU-host runtime).
Judged against `PREREG.md` (blob `d983111828355c30e2ab7bb5884a6b371624ff53`, embedded in the analysis notebook
before it read any label). Raw outputs are in `data/`; `data/analysis_summary.json` holds every number below.

## Verdict: PARTIAL, failed condition (d)

| | Condition | Result | |
|---|---|---|---|
| (a) | Coverage over ≥ 200 injection replicates ≥ 85% | 97.0% over 200 replicates | PASS |
| (b) | Pooled Redux 95% interval covers 181 | estimate 146.3, interval [102.3, 796.9] | PASS |
| (c) | Union misses ≥ 20% of the 181 Redux key errors | union finds 100, misses 44.8% | PASS |
| (d) | \|estimate − truth\| ≤ ½ \|union − truth\| on Redux and on average over replicates | Redux 34.7 vs 40.5 (passes); replicates 94.9 vs 23.0 (fails) | FAIL |

Under PREREG §2 (unchanged from v1), a single failure among (b)–(d) with (a) passing is a partial result, named
and not upgraded to GO.

## What happened, in plain words

v2 fixed what sank v1. With modern judges and a fixed budget of 285 flags each, the detectors behaved like
real auditors: together they checked 742 items, found 100 of the 181 key errors and missed 45%. So this time
there was something real for the estimator to estimate.

The estimator was honest but not useful. Its 95% intervals contained the truth in 97% of the injection
replicates and on the real benchmark, but only because they were very wide: the median interval was 681 items
wide on a true count of 266, and the Redux interval ran from 102 to 797. Its point estimate swung widely
between replicates (median 300, 10th–90th percentile 249–480, maximum 1,914) and overshot the truth in 75% of
them. On average it was 95 errors away from the truth, while simply counting what the detectors found was 46
away. It beat the raw count in only 58% of replicates. On the real benchmark it moved in the right direction
(100 found, estimate 146, truth 181) but still undercounted, and the model fit was poor (p = 0.002).

This is the failure the software-inspection literature warned about (Briand et al. 2000): with only a few
inspectors of very uneven skill, capture–recapture estimates become unstable. Here there were four detectors,
with recall from 10% to 46%.

## Per detector (Redux, budget of 285 flags each)

| Detector | Key errors caught (of 181) | Recall | Precision |
|---|---|---|---|
| D1 `google/gemma-4-31B-it` | 71 | 39.2% | 25.1% |
| D2 `Qwen/Qwen3.8-27B-FP8` (thinking off) | 38 | 21.0% | 13.5% |
| D3 `mistralai/Mistral-Small-4-119B-2603-NVFP4` | 19 | 10.5% | 6.7% |
| D4 confident learning on D1–D3 | 84 | 46.4% | 29.8% |

Confident learning was the best single detector on both recall and precision. Unscored rates were 0.00%,
0.11% and 0.00%.

## Other pre-registered readings

- **Errors no detector found (Redux):** 46.3, interval [2.3, 696.9].
- **Audit cost:** checking the 742 flagged items finds 100 key errors (v1: 4,145 items for 171).
- **Secondary estimators (Redux):** Mh jackknife 120.2 [100.0, 127.8]; Chao lower bound 107.6 [102.8, 121.0].
  Three of six Chapman pairs fall below the union count, a sign of strong positive dependence between pairs.
- **Selected interactions:** Gemma:cleanlab, Qwen:Mistral, Gemma:Qwen.
- **Supercategories (secondary):** the interval covers the truth in STEM, humanities and other; social sciences
  (16 true errors, 8 found) collapses to [8, 8] and misses.
- **Replicates:** mean union 219.9 of 266; no unstable fits; median goodness-of-fit p 0.07.

## Secondary and exploratory (not part of the verdict)

**Natural verdicts, no budget.** Had each judge flagged every item it called INCORRECT, Gemma would flag 613
(recall 55.8%, precision 16.6%), Qwen 772 (55.8%, 13.2%) and Mistral 1,452 (53.0%, 6.7%). Their union would
flag 1,869 items and catch 125 of 181, so even used naturally, current open LLM judges miss about 31% of known
MMLU key errors. Mistral-Small-4's written verdict was unreadable on 22% of real items (its first-token score
was always available); that is why the smoke check's 5% rule was moved to the unscored rate, see deviations.

**Planted errors are much easier than real ones.** On injected key swaps, the judges' natural verdict said
INCORRECT 91% (Gemma), 91% (Qwen) and 75% (Mistral) of the time, against about 55% recall on the real errors.
The calibration arm therefore under-represents how hard real errors are to see, which is one reason the method
looks better calibrated there than it is useful on Redux.

## Deviations from the plan

All environment or reporting fixes; none touched `PREREG.md`, a threshold or the flag rule.

1. The judge notebooks' version check still expected v1's vLLM 0.11 / transformers 4; corrected to the pinned
   0.31 / 5 (the installed versions were right).
2. The smoke check carried v1's rule that stops on unreadable verdict text. In v2 the verdict is secondary and
   the PREREG (§4) defines failure by the unscored rate, so the 5% limit was moved to unscored items. Mistral's
   first chunk had 26.8% unreadable verdicts and 0.0% unscored; it was restarted and resumed from Drive.

## Conclusion across v1 and v2

Capture–recapture over automated label-error detectors is well calibrated but, with three or four detectors of
uneven skill, too imprecise to beat simply counting what the detectors found. v1 failed because old judges
flagged most of the benchmark, so nothing was left to estimate. v2 created the right conditions and the
estimator still did not earn its keep. The project stops here. Findings worth reporting: current LLM judges
miss about a third of known MMLU key errors even without a budget; confident learning out-performs LLM judges
as a label-error detector; planted key swaps are far easier to detect than real errors, which limits injection
as a calibration test; and pre-registration with a smoke check caught two pipeline faults before they could
affect a result.
