"""Prompt construction, option shuffling, work-list building and verdict parsing.

This module holds everything a judge notebook needs that is not vLLM itself, so
that the fiddly parts -- the per-item option permutation, the choice of which 2
of 3 wrong keys to score, and verdict parsing -- can be tested on a laptop
without a GPU. `test_detectors.py` does exactly that.

Nothing here reads `error_type`. The work list a judge runs is built from the
questions, the options and the keys only, so a judge notebook never touches the
expert labels. Truth is joined to detector output in the analysis notebook,
after PREREG.md is committed.

See PREREG.md sections 3, 7, 9a.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict

import numpy as np

MASTER_SEED = 20261006
LETTERS = ("A", "B", "C", "D")

# row kinds in the work list
REDUX = "redux"          # the item's real key, shuffled order      (5,700 per judge)
INJECT = "inject"        # an `ok` item with a wrong key, shuffled  (10,660 per judge)
ORIGINAL = "original"    # the real key, ORIGINAL order, 600 items  (section 9a)


# --------------------------------------------------------------------------
# per-item randomisation, fixed by seed
# --------------------------------------------------------------------------


def option_permutation(item_index: int) -> np.ndarray:
    """The fixed permutation of the 4 options for this item.

    Drawn from (MASTER_SEED, item_index) so it is identical for every judge,
    for the confident-learning pass, for injection scoring and for every
    replicate. `perm[new_position] = old_position`.
    """
    rng = np.random.default_rng([MASTER_SEED, int(item_index)])
    return rng.permutation(4)


def scored_wrong_keys(item_index: int, answer: int) -> list[int]:
    """The 2 of the 3 wrong options this item is scored against (PREREG 7).

    Drawn at random per item rather than by position, so no option position is
    favoured. Returned as ORIGINAL option indices, sorted for stability.
    """
    wrong = [i for i in range(4) if i != int(answer)]
    rng = np.random.default_rng([MASTER_SEED, int(item_index), 1])
    picked = rng.choice(len(wrong), size=2, replace=False)
    return sorted(wrong[i] for i in picked)


def apply_permutation(options: list[str], key_index: int, perm: np.ndarray):
    """Reorder the options and move the key with its text.

    Returns (reordered_options, new_key_index). The caller labels the reordered
    options A-D in order, so the claimed key is named by its new letter and the
    original A-D positions carry no information.
    """
    reordered = [options[i] for i in perm]
    new_key = int(np.where(perm == int(key_index))[0][0])
    return reordered, new_key


# --------------------------------------------------------------------------
# the work list
# --------------------------------------------------------------------------


@dataclass
class Row:
    """One prompt a judge must answer."""

    row_id: str
    kind: str
    item_index: int
    subject: str
    claimed_key_original: int   # which ORIGINAL option is being claimed correct
    shuffled: bool

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


def build_work_list(items: list[dict], original_order_sample: set[int]) -> list[Row]:
    """Every prompt one judge runs, in a fixed order.

    `items` must be the pinned dataset in a fixed order, each dict carrying
    `subject`, `answer` and `choices`. `original_order_sample` holds the item
    indices in the 600-item section 9a sample.

    Per judge this is 5,700 redux rows + 2 x (number of `ok` items) inject rows
    + 600 original-order rows. The `ok` set is passed in by the caller as
    `item["is_ok"]`; see the note in build_inject_rows about why that is the one
    label the work list depends on.
    """
    rows: list[Row] = []
    for idx, item in enumerate(items):
        rows.append(
            Row(f"redux:{idx}", REDUX, idx, item["subject"], int(item["answer"]), True)
        )
    rows.extend(build_inject_rows(items))
    for idx in sorted(original_order_sample):
        item = items[idx]
        rows.append(
            Row(f"original:{idx}", ORIGINAL, idx, item["subject"], int(item["answer"]), False)
        )
    return rows


def build_inject_rows(items: list[dict]) -> list[Row]:
    """The injection-scoring rows: each `ok` item against 2 of its 3 wrong keys.

    This is the one place the work list depends on a label, because only `ok`
    items can carry an injected error -- flipping the key of an item that is
    already wrong would not give a known truth. It uses `is_ok` and nothing
    else: no error type, no correct_answer, no expert flag. The analysis
    notebook is still the only place truth is joined to detector output.
    """
    rows: list[Row] = []
    for idx, item in enumerate(items):
        if not item.get("is_ok"):
            continue
        for wrong in scored_wrong_keys(idx, int(item["answer"])):
            rows.append(
                Row(f"inject:{idx}:{wrong}", INJECT, idx, item["subject"], wrong, True)
            )
    return rows


def original_order_sample(items: list[dict], key_error_indices: list[int], n_total: int = 600) -> list[int]:
    """The section 9a sample: all key errors plus a stratified top-up.

    Built in the analysis notebook (it needs the expert labels to know which
    items are key errors) and handed to the judge notebooks as a plain list of
    indices, so a judge still never sees a label.
    """
    chosen = set(int(i) for i in key_error_indices)
    remaining = n_total - len(chosen)
    if remaining <= 0:
        return sorted(chosen)
    pool: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        if idx in chosen:
            continue
        pool.setdefault(item["supercategory"], []).append(idx)
    sizes = {k: len(v) for k, v in pool.items()}
    total = sum(sizes.values())
    if total == 0:
        return sorted(chosen)
    remaining = min(remaining, total)
    # proportional allocation with largest-remainder rounding, so the top-up is
    # stratified by supercategory and the total is exact
    exact = {k: remaining * n / total for k, n in sizes.items()}
    take = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: (-(exact[k] - take[k]), k)):
        if sum(take.values()) >= remaining:
            break
        if take[k] < sizes[k]:
            take[k] += 1
    rng = np.random.default_rng([MASTER_SEED, 9])
    for k in sorted(take):
        n = min(take[k], sizes[k])
        if n:
            picked = rng.choice(len(pool[k]), size=n, replace=False)
            chosen.update(pool[k][i] for i in picked)
    return sorted(chosen)


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


JUDGE_INSTRUCTION = (
    "You are checking a multiple-choice exam question for errors in its answer key.\n"
    "Decide whether the claimed answer is the correct answer to the question.\n"
    "Answer with the verdict first, in exactly this format:\n"
    "VERDICT: CORRECT\n"
    "REASON: <at most one short sentence>\n"
    "or\n"
    "VERDICT: INCORRECT\n"
    "REASON: <at most one short sentence>"
)


def judge_prompt(question: str, options: list[str], claimed_letter: str) -> str:
    """The judge prompt. Identical template for every model (PREREG 3)."""
    lines = [JUDGE_INSTRUCTION, "", f"Question: {question.strip()}", ""]
    for letter, text in zip(LETTERS, options):
        lines.append(f"{letter}. {str(text).strip()}")
    lines += ["", f"Claimed answer: {claimed_letter}", ""]
    return "\n".join(lines)


def answer_prompt(question: str, options: list[str]) -> str:
    """The confident-learning prompt: answer the question, no key shown."""
    lines = ["Answer the multiple-choice question. Reply with one letter only.", "",
             f"Question: {question.strip()}", ""]
    for letter, text in zip(LETTERS, options):
        lines.append(f"{letter}. {str(text).strip()}")
    lines += ["", "Answer:"]
    return "\n".join(lines)


def render_row(row: Row, item: dict) -> tuple[str, dict]:
    """Build the prompt for one work row, plus the metadata needed to read it back.

    Returns (prompt_text, meta). `meta["perm"]` records the permutation used, so
    the analysis notebook can map letters back to original option identities.
    """
    options = list(item["choices"])
    claimed = int(row.claimed_key_original)
    if row.shuffled:
        perm = option_permutation(row.item_index)
        options, claimed_new = apply_permutation(options, claimed, perm)
    else:
        perm = np.arange(4)
        claimed_new = claimed
    prompt = judge_prompt(item["question"], options, LETTERS[claimed_new])
    return prompt, {"perm": [int(x) for x in perm], "claimed_letter": LETTERS[claimed_new]}


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


_VERDICT_RE = re.compile(r"VERDICT\s*[:\-]?\s*\**\s*(CORRECT|INCORRECT)", re.I)
_BARE_RE = re.compile(r"\b(INCORRECT|CORRECT)\b", re.I)

CORRECT = "CORRECT"
INCORRECT = "INCORRECT"
UNPARSED = "unparsed"


def parse_verdict(text: str) -> str:
    """Read a judge's verdict. Anything unrecognisable is `unparsed`.

    An unparsed response counts as NOT flagged (PREREG 3), and the unparsed
    rate is reported per model; above 5% the model is reported as broken rather
    than treated as a detector that flags nothing.

    Note the order: INCORRECT is checked before CORRECT in the bare fallback,
    because "INCORRECT" contains "CORRECT" as a substring.
    """
    if not text:
        return UNPARSED
    m = _VERDICT_RE.search(text)
    if m:
        return m.group(1).upper()
    m = _BARE_RE.search(text)
    if m:
        return m.group(1).upper()
    return UNPARSED


def is_flag(verdict: str) -> bool:
    """A detector flags the item when it says the claimed key is INCORRECT."""
    return verdict == INCORRECT


# --------------------------------------------------------------------------
# the automatic smoke check (PREREG 10, 12)
# --------------------------------------------------------------------------

MAX_UNPARSED = 0.05  # PREREG 12: above this the model is reported as broken


@dataclass
class SmokeResult:
    """The verdict of the after-first-chunk check. `ok` False means stop."""

    ok: bool
    unparsed_rate: float
    prompts_per_s: float
    projected_hours: float
    budget_hours: float
    reasons: list[str]
    message: str


def smoke_check(
    verdicts: list[str],
    session_elapsed_s: float,
    generate_elapsed_s: float,
    n_done: int,
    n_total: int,
    budget_hours: float,
    label: str = "",
    max_unparsed: float = MAX_UNPARSED,
) -> SmokeResult:
    """Decide, after the first chunk, whether this notebook may keep going.

    Two pre-registered stop conditions, both fixed in advance and neither
    depending on any label:

    * unparsed rate above 5% -- PREREG 12 says the model is then reported as
      broken, not quietly treated as a detector that flags nothing;
    * projected session hours past this notebook's share of the GPU budget --
      PREREG 10.

    The projection is in *session* hours, not GPU-busy hours, because Colab
    bills session time: it is the time already spent plus the time the
    remaining rows will take at the measured rate.
    """
    n = max(len(verdicts), 1)
    unparsed_rate = sum(v == UNPARSED for v in verdicts) / n
    rate = n_done / generate_elapsed_s if generate_elapsed_s > 0 else 0.0
    remaining = max(n_total - n_done, 0)
    projected_hours = (
        (session_elapsed_s + remaining / rate) / 3600.0 if rate > 0 else float("inf")
    )

    reasons = []
    if unparsed_rate > max_unparsed:
        reasons.append(
            f"unparsed rate {unparsed_rate:.1%} is above the {max_unparsed:.0%} limit "
            f"(PREREG 12). The prompt or the parser does not fit this model. "
            f"Do not treat it as a detector that flags nothing."
        )
    if projected_hours > budget_hours:
        reasons.append(
            f"projected {projected_hours:.2f} session hours is past this notebook's "
            f"budget of {budget_hours:.2f} (PREREG 10) at {rate:.1f} prompts/s."
        )

    head = f"SMOKE CHECK {label}".strip()
    body = (
        f"  rows done        {n_done} of {n_total}\n"
        f"  unparsed rate    {unparsed_rate:.1%}  (limit {max_unparsed:.0%})\n"
        f"  prompts/s        {rate:.1f}\n"
        f"  projected hours  {projected_hours:.2f}  (budget {budget_hours:.2f})"
    )
    if reasons:
        message = (
            f"{head}: STOP\n{body}\n\n"
            + "\n".join(f"  * {r}" for r in reasons)
            + "\n\n  Nothing is lost: what is on Drive is kept and the notebook resumes.\n"
            "  Report this before changing anything. Do not retune the prompt or the\n"
            "  parser to make the number go away without saying so."
        )
    else:
        message = f"{head}: PASS, continuing automatically\n{body}"
    return SmokeResult(
        ok=not reasons,
        unparsed_rate=unparsed_rate,
        prompts_per_s=rate,
        projected_hours=projected_hours,
        budget_hours=budget_hours,
        reasons=reasons,
        message=message,
    )
