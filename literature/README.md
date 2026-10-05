# Literature

Novelty status and verified closest work: `NOVELTY.md`. Raw novelty-check notes: `novelty_checks/`.

Verification rule (CLAUDE.md): nothing goes in this table until the record has been opened. Every row below was
checked on 2026-10-06 against the source named in the "verified via" column — arXiv API (`export.arxiv.org/api/query`),
Crossref (`api.crossref.org/works/<doi>`), OpenAlex, the ACL Anthology page, or the publisher/CRAN page itself.
"Read" says how much of the paper has actually been read so far, so the reading state is never overstated.

## Core reading list (phase 0)

| # | Paper | Authors (verified) | Venue (verified) | Link | Verified via | Read | What we take from it |
|---|---|---|---|---|---|---|---|
| 1 | Are We Done with MMLU? | Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, Claire Barale, Robert McHardy, Joshua Harris, Jean Kaddour, Emile van Krieken, Pasquale Minervini (16 authors) | NAACL 2025, Long Papers, pp. 5069–5096, doi 10.18653/v1/2025.naacl-long.262; arXiv 2406.04127 (v1 2024-06-06, v2 2025-01-10) | https://arxiv.org/abs/2406.04127 | arXiv API + Crossref | abstract + full dataset card (paper body not yet read) | Our ground truth. The 7-value `error_type` taxonomy and the annotation protocol (exact-match web search for the question, then expert judgement, `expert` when still unsure). The paper's headline "57% of Virology is in error" reproduces exactly on the pinned revision we pulled (virology: 57 of 100 not `ok`), which is a sanity check that we are reading the right data. |
| 2 | Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks | Curtis G. Northcutt, Anish Athalye, Jonas Mueller | NeurIPS 2021 Datasets & Benchmarks (journal_ref on arXiv); arXiv 2103.14749 (v1 2021-03-26, v4 2021-11-07) | https://arxiv.org/abs/2103.14749 | arXiv API (journal_ref field) | abstract | Confident learning as detector D4, and the "at least 3.3%" lower-bound framing that this project is trying to replace with an estimate plus a CI. Confirms NOVELTY.md's correction of Gemini (NeurIPS D&B, not JAIR). |
| 3 | Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance | Omer Nahum, Nitay Calderon, Orgad Keller, Idan Szpektor, Roi Reichart | EMNLP 2025 Main, pp. 26770–26797, doi 10.18653/v1/2025.emnlp-main.1360 | https://aclanthology.org/2025.emnlp-main.1360/ | Crossref + ACL Anthology page | abstract | The LLM-judge-ensemble detector design (D1–D3). They pool judge votes into one decision; we keep each judge as a separate capture occasion instead, which is exactly the difference the novelty claim rests on. |
| 4 | Auditing LLM Benchmarks with Item Response Theory | Sander Land, Daniel M. Bikel | arXiv 2605.30504 (v1 2026-05-28, v2 2026-08-28); arXiv comment: "Accepted at EMNLP 2026" | https://arxiv.org/abs/2605.30504 | arXiv API | abstract | The IRT-residual detector (optional D5). Their indicator reaches 95% precision in the top 200 items across seven benchmarks using responses from 114 models. Associated data: https://huggingface.co/datasets/Writer/IRT-mislabeled-items. Note the dependence risk: an IRT residual over model responses and an LLM judge may both key on the same "strong models agree on a non-key option" signal. |
| 5 | Fantastic Bugs and Where to Find Them in AI Benchmarks | Sang Truong, Yuheng Tu, Michael Hardy, Anka Reuel, Zeyu Tang, Jirayu Burapacheep, Jonathan Perera, Chibuike Uwakwe, Ben Domingue, Nick Haber, Sanmi Koyejo (11 authors) | arXiv 2511.16842 (2025-11-20). No journal_ref and no venue in the arXiv comment — NOVELTY.md is right that Gemini's "NeurIPS D&B" is unverified. | https://arxiv.org/abs/2511.16842 | arXiv API | abstract | Response-matrix statistics plus an LLM judge to prioritise items for human review. Closest in spirit on the detector side; still prioritisation, with no estimate of what review never reaches. |
| 6a | Capture–recapture in software inspections after 10 years research — theory, evaluation and application | Håkan Petersson, Thomas Thelin, Per Runeson, Claes Wohlin | Journal of Systems and Software 72(3):249–264, July 2004, doi 10.1016/s0164-1212(03)00090-6 | https://doi.org/10.1016/s0164-1212(03)00090-6 | Crossref | **metadata only — closed access, no abstract in Crossref, no open-access copy (OpenAlex: `oa_status: closed`; Lund University record https://lup.lub.lu.se/record/277230 is not open)** | The main precedent for the transfer. Not yet read: see the open question in `STATUS.md`. Confirms NOVELTY.md's correction of Gemini (Petersson is first author, not Wohlin). |
| 6b | A comprehensive evaluation of capture-recapture models for estimating software defect content | Lionel C. Briand, Khaled El Emam, Bernd G. Freimut, Oliver Laitenberger | IEEE Transactions on Software Engineering 26(6):518–540, June 2000, doi 10.1109/32.852741 | https://doi.org/10.1109/32.852741 | Crossref + OpenAlex (abstract) | abstract (green OA full text at http://publica.fraunhofer.de/documents/PX-45003.html) | **The most important methodological warning we have.** On real inspection data: estimator accuracy is strongly driven by the *number of inspectors*; with too few inspectors no model is accurate and underestimation may be substantial. They recommend a model in which defects have different detection probabilities (Mh) with the **Jackknife** estimator. This goes straight into the pre-registration. Confirms NOVELTY.md's correction of Gemini (Briand 2000, not "El Emam 2001"). |
| 7a | Estimating the Population Size for Capture-Recapture Data with Unequal Catchability | Anne Chao | Biometrics 43(4):783–791, Dec 1987, doi 10.2307/2531532 | https://doi.org/10.2307/2531532 | Crossref | metadata only | The unequal-catchability (Mth) estimator. This is the "Chao" the plan should name where it currently says "Chao (1987) lower bound". |
| 7b | Nonparametric estimation of the number of classes in a population | Anne Chao | Scandinavian Journal of Statistics 11:265–270, 1984. **No DOI** (not in Crossref) | https://openalex.org/ record via OpenAlex search | OpenAlex (title, author, journal, volume, pages, year) | metadata only | The actual Chao *lower bound* (Chao1), the robust floor under heterogeneous detectability. Cite with volume/pages, not a DOI. |
| 7c | The multiple recapture census for closed populations and incomplete 2^k contingency tables | Stephen E. Fienberg | Biometrika 59(3):591–603, 1972, doi 10.1093/biomet/59.3.591 | https://doi.org/10.1093/biomet/59.3.591 | Crossref | metadata only | The log-linear multiple-systems model — our pre-registered primary estimator. |
| 7d | A Sample Coverage Approach to Multiple-System Estimation with Application to Census Undercount | Anne Chao, P. K. Tsay | Journal of the American Statistical Association 93(441):283–293, March 1998, doi 10.1080/01621459.1998.10474109 | https://doi.org/10.1080/01621459.1998.10474109 | Crossref | metadata only — closed access | Sample-coverage estimator. **Not implemented and dropped from the pre-registration**: we cannot implement an estimator from a paper we have not read, and guessing its form would be worse than leaving it out. Revisit if access appears. |
| 7e | The Rcapture Package: Loglinear Models for Capture-Recapture in R | Sophie Baillargeon, Louis-Paul Rivest | Journal of Statistical Software 19(5), 2007, doi 10.18637/jss.v019.i05 | https://doi.org/10.18637/jss.v019.i05 | Crossref + CRAN | metadata only | Reference implementation to cross-check our Python estimators against. CRAN page opened: Rcapture 1.4-4, published 2022-05-04, authors Louis-Paul Rivest [aut, cre], Sophie Baillargeon [aut]. |

## Added during phase 0 (not in the original plan)

| Paper | Authors | Venue | Link | Verified via | Why it was added |
|---|---|---|---|---|---|
| Small Sample Inference for Two-way Capture Recapture Experiments | Louis-Paul Rivest, Mamadou Yauck | arXiv 2307.11705 (v1 2023-07-21, v2 2024-02-28); no journal_ref | https://arxiv.org/abs/2307.11705 | arXiv API | Treats the small-sample Petersen/Chapman case and gives the Chapman estimator's properties, which we need because our Chapman step has no verified primary source (see below). |


## Added during phase 1 (estimators)

| Paper | Authors (verified) | Venue (verified) | Link | Verified via | Why it is here |
|---|---|---|---|---|---|
| Estimation of the size of a closed population when capture probabilities vary among animals | K. P. Burnham, W. S. Overton | Biometrika 65(3):625–633, 1978, doi 10.1093/biomet/65.3.625 | https://doi.org/10.1093/biomet/65.3.625 | Crossref | The Mh jackknife estimator the author asked for. Our `crc.py` does **not** transcribe its coefficients: it re-derives them by Richardson extrapolation and `test_crc.py` checks the result against the published closed forms for orders 1–3 (300 cases, exact arithmetic). |
| Robust Estimation of Population Size When Capture Probabilities Vary Among Animals | K. P. Burnham, W. S. Overton | Ecology 60(5):927–936, Oct 1979, doi 10.2307/1936861 | https://doi.org/10.2307/1936861 | Crossref | The sequential test that picks the jackknife order. |
| Statistical inference from capture data on closed animal populations | David L. Otis, Kenneth P. Burnham, Gary C. White, David R. Anderson | Wildlife Monographs 62:3–135, 1978. **No DOI** (not in Crossref) | OpenAlex record | OpenAlex (title, authors, journal, volume, pages, year) | The source of the M0 / Mt / Mh / Mb / Mth model names used throughout `PREREG.md` and the simulation scenarios. |
| Confident Learning: Estimating Uncertainty in Dataset Labels | Curtis G. Northcutt, Lu Jiang, Isaac L. Chuang | Journal of Artificial Intelligence Research (JAIR), 2021 (arXiv journal_ref); arXiv 1911.00068 (v1 2019-10-31, v5 2022-08-22) | https://arxiv.org/abs/1911.00068 | arXiv API | The method behind `cleanlab.filter.find_label_issues`, which is detector D5. This is a **different paper** from the NeurIPS D&B one in row 2; the pre-registration cites this one for the method and row 2 for the "at least 3.3%" framing. |
| Small Sample Inference for Two-way Capture Recapture Experiments | Louis-Paul Rivest, Mamadou Yauck | arXiv 2307.11705 (v1 2023-07-21, v2 2024-02-28); no journal_ref | https://arxiv.org/abs/2307.11705 | arXiv API | Already added in phase 0; it is also how `PREREG.md` cites the Chapman estimator, since Chapman (1951) could not be located. |

## References named in PLAN.md that are NOT yet verified

- **Chapman (1951)**, "Some properties of the hypergeometric distribution with applications to zoological sample
  censuses", University of California Publications in Statistics 1:131–160. Not in Crossref, not in OpenAlex, no
  open copy found. The Chapman (bias-corrected Lincoln–Petersen) estimator stays in the design, but `PREREG.md`
  must cite it through a source we have actually opened — Rivest & Yauck (arXiv 2307.11705) or Briand et al. (2000) —
  rather than through an unopened 1951 university publication.

## Supporting sources opened in phase 0

| Source | What it gave us | Pinned identifier |
|---|---|---|
| `edinburgh-dawg/mmlu-redux-2.0` on Hugging Face | The ground truth; all counts in `experiments/kill_v1/DATA.md` | revision `372ea425445d51e1ba1188c56e5e893f8138621f` |
| `hendrycks/test` `categories.py` on GitHub | The official MMLU subject → supercategory map used for our 4 strata | https://raw.githubusercontent.com/hendrycks/test/master/categories.py |
| Hugging Face model API + model cards | Detector model upload dates and training cutoffs (contamination control) | see `experiments/kill_v1/DATA.md` |

PDFs, where we have them, go in `papers_local/` (gitignored).
