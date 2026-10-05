# How many label errors are left? Capture–recapture for benchmark audits

Private working repository. Not public until there is a result worth showing.

Every automated label-error detector misses some errors, so the counts reported today are lower bounds. Ecologists face the same problem when counting animals: no single survey sees every animal, but the overlap between two or more surveys tells you how many nobody saw. This project treats each error detector (LLM judges from different model families, confident learning, cross-model disagreement) as one "survey", checks the items it flags, and uses the overlap between detectors to estimate the total number of label errors in a benchmark, including those no detector found, with a confidence interval.

| Where | What |
|---|---|
| `CLAUDE.md` | Brief and rules for the Claude Code session that does the work |
| `PLAN.md` | Phases, kill experiment, GO / NO-GO thresholds, compute |
| `docs/IDEA.md` | The idea in plain language |
| `literature/` | Novelty status (NOT FOUND in 3 checks), verified closest work, reading list |
| `experiments/kill_v1/` | Pre-registration, code, Colab notebooks, executed runs |
