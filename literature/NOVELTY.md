# Novelty check: capture–recapture for the total number of label errors in a benchmark

Idea: treat each automated label-error detector (LLM judges from different families, confident learning, IRT residuals,
cross-model disagreement) as one capture occasion; use multiple-systems estimation (Lincoln–Petersen, log-linear models
with list dependence, Chao's lower bound / Mh models) to estimate the TOTAL number of label errors in a benchmark,
including errors no detector found, with a CI. Validate on expert re-annotations (MMLU-Redux) and injected errors.

## Verdict: NOT FOUND (three independent checks)
1. arXiv + OpenReview agent (2026-10-05): NOT FOUND.
2. Semantic Scholar + web agent (2026-10-05), incl. all 773 citers of Northcutt 2021: NOT FOUND.
3. Gemini Deep Research (2026-10-06): NOT FOUND.

## Closest work (all references verified 2026-10-06; Gemini errors corrected)
| Paper | Verified venue | What it does | Difference |
|---|---|---|---|
| Northcutt et al., "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks", arXiv 2103.14749 | NeurIPS 2021 Datasets & Benchmarks (Gemini wrongly said JAIR) | Confident learning flags label errors; reports counts ("at least 3.3%") | Single detector, lower bound only |
| Gema et al., "Are We Done with MMLU?", arXiv 2406.04127 | NAACL 2025 (doi 10.18653/v1/2025.naacl-long.262) | MMLU-Redux: expert re-annotation | Ground truth for validation, no estimator |
| Nahum et al., "Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance" | EMNLP 2025 (doi 10.18653/v1/2025.emnlp-main.1360) | LLM-judge ensemble flags errors | Pools votes; no estimate of undetected errors |
| Land et al., "Auditing LLM Benchmarks with Item Response Theory", arXiv 2605.30504 | accepted at EMNLP 2026 (arXiv comment) | 4PL IRT over 114 models surfaces mislabels | One detector; no total |
| Truong et al., "Fantastic Bugs and Where to Find Them in AI Benchmarks", arXiv 2511.16842 | arXiv (Gemini's "NeurIPS D&B" NOT verified) | Response-matrix signals + LLM judge prioritise items for review | Prioritisation, no population estimate |
| Petersson, Thelin, Runeson & Wohlin (2004), "Capture–recapture in software inspections after 10 years research", J. Systems and Software, doi 10.1016/s0164-1212(03)00090-6 | JSS 2004 (Gemini wrongly gave Wohlin as first author) | Survey of M0/Mt/Mh/Mth for counting remaining software defects from inspector overlap | Human inspectors, software defects — **the main precedent: the transfer, not the statistics, is new** |
| Briand, El Emam, Freimut & Laitenberger (2000), "A comprehensive evaluation of capture-recapture models for estimating software defect content", IEEE TSE, doi 10.1109/32.852741 | IEEE TSE 2000 (Gemini wrongly said El Emam 2001) | Robustness of estimators for inspection data | Same as above |
| Difallah, Filatova & Ipeirotis (2018), "Demographics and Dynamics of Mechanical Turk Workers", WSDM, doi 10.1145/3159652.3159661 | WSDM 2018 | Capture–recapture for crowd-worker population size | Different population |

## Main risks identified before the experiments
1. **Detector dependence** (all LLMs miss the same subtle errors) → underestimate. Need ≥3 detectors from different families and log-linear interaction terms; Chao lower bound as a robust floor.
2. **False-positive flags** → captures must be *verified* errors: verify flagged items only (expert labels in MMLU-Redux), or model false positives explicitly.
3. **Heterogeneous detectability** (typos vs subtle reasoning errors) → Mh / covariate (Huggins) models.
