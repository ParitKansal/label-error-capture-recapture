"""Write the Colab notebooks for the kill experiment.

Each notebook embeds the src files it needs via %%writefile, so a notebook is
self-contained once uploaded: no git clone, no Drive copy of the source.

    python make_notebook.py --out ../notebooks

Produces:
  judge_<model>.ipynb   x4  one per model family. GPU (A100). Runs that model's
                            16,360 judge prompts, its option-probability pass
                            for confident learning, and its share of the
                            600-item original-order check.
  modern_arm.ipynb          GPU. Exploratory (PREREG 9b): two large models,
                            Redux pass only. First in the cut order.
  analysis.ipynb            CPU High-RAM. Joins truth, runs the estimators and
                            the 200+ injection replicates.

Every GPU notebook writes JSONL to Drive after each chunk and skips row ids it
already has, so a disconnect costs one chunk, not a run. None of them reads
`error_type`: truth is joined only in the analysis notebook, after PREREG.md is
committed (PREREG 9).
"""

from __future__ import annotations

import argparse
import json
import os

DRIVE_ROOT = "/content/drive/MyDrive/label_error_crc/kill_v1"
REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"

# PREREG section 3. (id, short name, bf16 weights, note)
JUDGES = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "llama3", "gated: accept the licence and set HF_TOKEN"),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral", ""),
    ("Qwen/Qwen2-7B-Instruct", "qwen2", ""),
    ("microsoft/Phi-3-medium-4k-instruct", "phi3", "14B, the slowest of the four"),
]

# PREREG section 9b, both verified on the Hub 2026-10-07.
MODERN = [
    ("Qwen/Qwen3.8-27B-FP8", "qwen38", "30.9 GB, FP8 has no native Ampere support"),
    ("google/gemma-4-31B-it-qat-w4a16-ct", "gemma4", "23.3 GB, W4A16 runs natively on Ampere"),
]


def _code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src.strip("\n").splitlines(keepends=True)}


def _md(src: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": src.strip("\n").splitlines(keepends=True)}


def _notebook(cells: list[dict], gpu: bool) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU" if gpu else "None",
            "colab": {"provenance": [], "machine_shape": "hm"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def _embed(src_dir: str, name: str) -> dict:
    with open(os.path.join(src_dir, name)) as fh:
        body = fh.read()
    return _code(f"%%writefile {name}\n{body}")


# --------------------------------------------------------------------------

SETUP = f'''
# Colab A100 40GB, Ampere (SM80). Pinned so a vLLM release cannot change the run.
!pip -q install "vllm==0.11.0" "huggingface_hub>=0.26" 2>&1 | tail -2
import os, torch
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0), "(8,0) = Ampere, no native FP8")
print("memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))

from google.colab import drive
drive.mount("/content/drive")
DRIVE = "{DRIVE_ROOT}"
os.makedirs(DRIVE, exist_ok=True)
print("drive ready:", DRIVE)
'''

DATA_CELL = f'''
# The pinned dataset. Downloads the 57 Arrow files at the exact revision in DATA.md.
# `is_ok` is the ONLY label used here, and only to decide which items can carry an
# injected error (PREREG 7). No error type, no correct_answer, no expert flag.
import json, urllib.parse, urllib.request, os
import pyarrow as pa, pyarrow.ipc as ipc

REVISION = "{REVISION}"
DATASET = "edinburgh-dawg/mmlu-redux-2.0"
CACHE = "/content/mmlu_redux"
os.makedirs(CACHE, exist_ok=True)

tree = json.load(urllib.request.urlopen(
    f"https://huggingface.co/api/datasets/{{DATASET}}/tree/{{REVISION}}?recursive=true", timeout=60))
paths = sorted(e["path"] for e in tree if e["type"] == "file" and e["path"].endswith(".arrow"))
assert len(paths) == 57, f"expected 57 arrow files, got {{len(paths)}}"

# supercategories from the official MMLU map, same source as DATA.md
src = urllib.request.urlopen(
    "https://raw.githubusercontent.com/hendrycks/test/master/categories.py", timeout=60).read().decode()
ns = {{}}
exec(compile(src, "categories.py", "exec"), ns)
SUB2CAT = {{s: c for s, subs in ns["subcategories"].items()
            for c, members in ns["categories"].items() if subs[0] in members}}

ITEMS = []
for path in paths:
    subject = path.split("/")[0]
    local = os.path.join(CACHE, subject + ".arrow")
    if not os.path.exists(local):
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/{{DATASET}}/resolve/{{REVISION}}/" + urllib.parse.quote(path),
            local)
    with pa.memory_map(local, "rb") as fh:
        cols = ipc.open_stream(fh).read_all().to_pydict()
    for i in range(len(cols["question"])):
        ITEMS.append({{
            "subject": subject,
            "supercategory": SUB2CAT[subject],
            "question": cols["question"][i],
            "choices": list(cols["choices"][i]),
            "answer": int(cols["answer"][i]),
            "is_ok": cols["error_type"][i] == "ok",
        }})

assert len(ITEMS) == 5700, len(ITEMS)
print(len(ITEMS), "items;", sum(i["is_ok"] for i in ITEMS), "ok items")
'''

WORKLIST_CELL = '''
import detectors as D

# The 600-item original-order sample (PREREG 9a) is built by the analysis
# notebook, which has the labels, and dropped on Drive as a plain index list.
# If it is not there yet, this notebook skips those rows and they can be run later.
sample_path = os.path.join(DRIVE, "original_order_sample.json")
if os.path.exists(sample_path):
    with open(sample_path) as fh:
        SAMPLE = set(json.load(fh))
else:
    SAMPLE = set()
    print("no original-order sample on Drive yet; skipping the 9a rows")

ROWS = D.build_work_list(ITEMS, SAMPLE)
from collections import Counter
print(Counter(r.kind for r in ROWS), "total", len(ROWS))
'''

RESUME_CELL = '''
# Resume: read back whatever this model already finished.
OUT = os.path.join(DRIVE, f"judge_{SHORT}.jsonl")
done = set()
if os.path.exists(OUT):
    with open(OUT) as fh:
        for line in fh:
            try:
                done.add(json.loads(line)["row_id"])
            except Exception:
                pass  # a line torn by a disconnect; it will simply be redone
print(f"{len(done)} rows already done, {len(ROWS) - len(done)} to go")
todo = [r for r in ROWS if r.row_id not in done]
'''

JUDGE_RUN_CELL = '''
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, dtype="bfloat16", gpu_memory_utilization=0.90,
          max_model_len=4096, enforce_eager=False)
params = SamplingParams(temperature=0.0, max_tokens=24)

CHUNK = 512
t0 = time.time()
for start in range(0, len(todo), CHUNK):
    chunk = todo[start:start + CHUNK]
    prompts, metas = [], []
    for row in chunk:
        text, meta = D.render_row(row, ITEMS[row.item_index])
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True))
        metas.append(meta)
    outs = llm.generate(prompts, params)
    with open(OUT, "a") as fh:
        for row, meta, out in zip(chunk, metas, outs):
            text = out.outputs[0].text
            fh.write(json.dumps({
                "row_id": row.row_id, "kind": row.kind, "item_index": row.item_index,
                "claimed_key_original": row.claimed_key_original,
                "perm": meta["perm"], "claimed_letter": meta["claimed_letter"],
                "raw": text[:200], "verdict": D.parse_verdict(text),
            }) + "\\n")
    done_n = start + len(chunk)
    rate = done_n / (time.time() - t0)
    print(f"{done_n}/{len(todo)}  {rate:.1f} prompts/s  "
          f"eta {(len(todo) - done_n) / max(rate, 1e-9) / 60:.1f} min", flush=True)

print("JUDGE PASS DONE")
'''

PROBS_CELL = '''
# Confident learning (D5): score the four options WITHOUT showing a key.
# One pass over 5,700 items. The probabilities do not depend on the claimed key,
# so this one pass serves the Redux analysis and every injection replicate.
# Options are in the same shuffled order; we map back to original identities here,
# so cleanlab sees option content, not letters.
import numpy as np
from vllm import SamplingParams

PROBS_OUT = os.path.join(DRIVE, f"probs_{SHORT}.jsonl")
done_p = set()
if os.path.exists(PROBS_OUT):
    with open(PROBS_OUT) as fh:
        for line in fh:
            try:
                done_p.add(json.loads(line)["item_index"])
            except Exception:
                pass

todo_p = [i for i in range(len(ITEMS)) if i not in done_p]
print(f"{len(todo_p)} items to score")

lp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
CHUNK = 512
for start in range(0, len(todo_p), CHUNK):
    idxs = todo_p[start:start + CHUNK]
    prompts, perms = [], []
    for i in idxs:
        item = ITEMS[i]
        perm = D.option_permutation(i)
        opts = [item["choices"][j] for j in perm]
        prompts.append(tok.apply_chat_template(
            [{"role": "user", "content": D.answer_prompt(item["question"], opts)}],
            tokenize=False, add_generation_prompt=True))
        perms.append(perm)
    outs = llm.generate(prompts, lp)
    with open(PROBS_OUT, "a") as fh:
        for i, perm, out in zip(idxs, perms, outs):
            top = out.outputs[0].logprobs[0] if out.outputs[0].logprobs else {}
            scores = {}
            for tid, info in top.items():
                t = info.decoded_token.strip().upper()
                if t in ("A", "B", "C", "D"):
                    scores[t] = max(scores.get(t, -1e9), info.logprob)
            vec = np.array([scores.get(L, -20.0) for L in ("A", "B", "C", "D")])
            vec = np.exp(vec - vec.max()); vec = vec / vec.sum()
            # map shuffled position -> original option index
            original = np.zeros(4)
            for new_pos, old_pos in enumerate(perm):
                original[old_pos] = vec[new_pos]
            fh.write(json.dumps({"item_index": i, "probs": original.tolist()}) + "\\n")
    print(f"{start + len(idxs)}/{len(todo_p)}", flush=True)

print("PROBABILITY PASS DONE")
'''

FINISH_CELL = '''
from google.colab import drive
drive.flush_and_unmount()
print("flushed. Close this session now -- Colab bills session time, not GPU use.")
'''


def judge_notebook(src_dir: str, model: str, short: str, note: str) -> dict:
    head = f"""
# Judge worker: `{model}`

PREREG.md section 3. This notebook runs **one** model family and writes to Drive as it goes.
{"**" + note + "**" if note else ""}

Runtime: **A100 GPU**. Run the four judge notebooks in **separate Colab sessions, in parallel**.
It resumes from whatever is already on Drive, so a disconnect costs one 512-row chunk.

Order of cells: setup, data, work list, resume, judge pass, probability pass, flush.
"""
    cells = [
        _md(head),
        _code(SETUP),
        _code(f'MODEL = "{model}"\nSHORT = "{short}"\nprint(MODEL)'),
        _embed(src_dir, "detectors.py"),
        _code(DATA_CELL),
        _code(WORKLIST_CELL),
        _code(RESUME_CELL),
        _md("## Judge pass\n\n16,360 prompts: 5,700 Redux + 10,660 injection scoring, plus the 600-item original-order rows if the sample is on Drive."),
        _code(JUDGE_RUN_CELL),
        _md("## Probability pass for confident learning (D5)\n\nSame session, weights already loaded."),
        _code(PROBS_CELL),
        _md("## Finish\n\nFlush Drive, then **close the session**."),
        _code(FINISH_CELL),
    ]
    return _notebook(cells, gpu=True)


MODERN_CELL = '''
# PREREG 9b. Exploratory, not a detector, FIRST in the cut order.
# Redux pass only: 5,700 prompts per model, shuffled options, same judge prompt.
import time, gc, torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

REDUX_ROWS = [r for r in ROWS if r.kind == D.REDUX]
params = SamplingParams(temperature=0.0, max_tokens=24)

for MODEL, SHORT, NOTE in MODERN_MODELS:
    OUT = os.path.join(DRIVE, f"modern_{SHORT}.jsonl")
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for line in fh:
                try: done.add(json.loads(line)["row_id"])
                except Exception: pass
    todo = [r for r in REDUX_ROWS if r.row_id not in done]
    print(f"=== {MODEL} ({NOTE}) -- {len(todo)} rows to go ===", flush=True)
    if not todo:
        continue
    try:
        tok = AutoTokenizer.from_pretrained(MODEL)
        llm = LLM(model=MODEL, gpu_memory_utilization=0.92, max_model_len=4096)
    except Exception as exc:
        # PREREG 9b: if a model will not load on this card, report it and carry on
        # with the other one. Do NOT substitute a model chosen after the fact.
        print(f"FAILED TO LOAD {MODEL}: {type(exc).__name__}: {exc}")
        with open(os.path.join(DRIVE, "modern_arm_failures.txt"), "a") as fh:
            fh.write(f"{MODEL}\\t{type(exc).__name__}\\t{exc}\\n")
        continue
    t0 = time.time()
    CHUNK = 256
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start:start + CHUNK]
        prompts, metas = [], []
        for row in chunk:
            text, meta = D.render_row(row, ITEMS[row.item_index])
            prompts.append(tok.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True))
            metas.append(meta)
        outs = llm.generate(prompts, params)
        with open(OUT, "a") as fh:
            for row, meta, out in zip(chunk, metas, outs):
                text = out.outputs[0].text
                fh.write(json.dumps({
                    "row_id": row.row_id, "item_index": row.item_index,
                    "claimed_key_original": row.claimed_key_original,
                    "raw": text[:200], "verdict": D.parse_verdict(text),
                }) + "\\n")
        n = start + len(chunk)
        rate = n / (time.time() - t0)
        print(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo)-n)/max(rate,1e-9)/60:.1f} min", flush=True)
    del llm, tok
    gc.collect(); torch.cuda.empty_cache()

print("MODERN ARM DONE")
'''


def modern_notebook(src_dir: str) -> dict:
    models = ",\n    ".join(f'("{m}", "{s}", "{n}")' for m, s, n in MODERN)
    head = """
# Exploratory: what do modern judges still miss?

PREREG.md section 9b. **Not part of GO and not a detector in the estimator.**
This arm is **first in the cut order** (PREREG 10): if the running GPU total threatens the
7 A100-hour stop, do not run this notebook.

Runtime: **A100 GPU**, one session, both models in sequence. Redux pass only (5,700 prompts each).

Two risks this notebook settles rather than assumes: the Colab A100 is 40 GB **Ampere (SM80)**,
which has no native FP8, so vLLM must use a weight-only path for the Qwen model, and its 30.9 GB of
weights leave under 9 GB for KV cache. If a model fails to load, the notebook records the failure and
continues with the other one -- it does **not** substitute a different model.
"""
    cells = [
        _md(head),
        _code(SETUP),
        _code(f"MODERN_MODELS = [\n    {models},\n]\nfor m in MODERN_MODELS: print(m[0])"),
        _embed(src_dir, "detectors.py"),
        _code(DATA_CELL),
        _code(WORKLIST_CELL),
        _code(MODERN_CELL),
        _code(FINISH_CELL),
    ]
    return _notebook(cells, gpu=True)


ANALYSIS_CELL = '''
# Join truth to detector output. THIS IS THE ONLY PLACE EXPERT LABELS ARE READ.
# Record the commit hash of PREREG.md first (PREREG 9, 13).
PREREG_COMMIT = ""  # <-- paste `git rev-parse HEAD:experiments/kill_v1/PREREG.md` here
assert PREREG_COMMIT, "record the pre-registration commit hash before running the analysis"

PRIMARY = {"wrong_groundtruth", "no_correct_answer", "multiple_correct_answers"}
EXPERT = "expert"

truth = []
for i, item in enumerate(ITEMS_FULL):
    et = item["error_type"]
    truth.append({
        "item_index": i,
        "subject": item["subject"],
        "supercategory": item["supercategory"],
        "error_type": et,
        "is_key_error": et in PRIMARY,
        "is_expert": et == EXPERT,
        "is_ok": et == "ok",
    })
n_key = sum(t["is_key_error"] for t in truth)
n_expert = sum(t["is_expert"] for t in truth)
print(f"{n_key} key errors, {n_expert} expert items excluded from primary truth")
assert n_key == 181, f"expected 181 key errors at the pinned revision, got {n_key}"
'''

REPLICATE_CELL = '''
# The 200+ injection replicates (PREREG 7). CPU only; parallel over seeds.
import numpy as np
from joblib import Parallel, delayed
import crc

MASTER_SEED = 20261006
N_REPLICATES = 200
PER_STRATUM = {"STEM": 86, "humanities": 60, "social sciences": 57,
               "other (business, health, misc.)": 63}  # 266 total, PREREG 7

ok_by_cat = {}
for t in truth:
    if t["is_ok"]:
        ok_by_cat.setdefault(t["supercategory"], []).append(t["item_index"])

def one_replicate(rep):
    rng = np.random.default_rng(MASTER_SEED + rep)
    injected = []
    for cat, n in PER_STRATUM.items():
        pool = ok_by_cat[cat]
        injected.extend(int(x) for x in rng.choice(pool, size=n, replace=False))
    injected = set(injected)
    # which wrong key each injected item carries, drawn from the 2 that were scored
    flipped_key = {}
    for i in sorted(injected):
        pair = detectors.scored_wrong_keys(i, ITEMS[i]["answer"])
        flipped_key[i] = int(rng.choice(pair))
    flags = build_flag_matrix(flipped_key)          # (5700, 5) 0/1
    is_error = np.zeros(len(ITEMS), dtype=bool)
    is_error[sorted(injected)] = True
    data = crc.CaptureData.from_flags(flags, is_error, DETECTOR_NAMES)
    est = crc.loglinear(data, n_boot=2000, seed=MASTER_SEED)
    return {
        "rep": rep, "truth": len(injected), "union": data.n_found,
        "n_hat": est.n_hat, "lo": est.lo, "hi": est.hi, "ok": est.ok,
        "gof_p": est.extra["gof_p"], "terms": est.extra["terms"],
        "covers": est.covers(len(injected)),
    }

results = Parallel(n_jobs=-1, verbose=5)(
    delayed(one_replicate)(r) for r in range(N_REPLICATES))
coverage = np.mean([r["covers"] for r in results])
print(f"empirical coverage over {N_REPLICATES} replicates: {coverage:.1%}  (GO needs >= 85%)")
'''


def analysis_notebook(src_dir: str) -> dict:
    head = """
# Analysis

Runtime: **CPU, High-RAM**. No GPU. Run this only after the judge notebooks have finished and
`PREREG.md` is committed.

This is the **only** notebook that reads `error_type`. It:
1. records the commit hash of `PREREG.md`;
2. builds the 600-item original-order sample and writes it to Drive for the judge notebooks (PREREG 9a);
3. joins truth to detector output;
4. runs confident learning (D5) from the stored option probabilities;
5. runs the estimators on the Redux arm;
6. runs the 200+ injection replicates and reports empirical coverage;
7. checks each of the four GO conditions and writes `RESULTS.md`.

Nothing in it may be retuned after looking at results. Anything not named in `PREREG.md` goes in the
exploratory section.
"""
    cells = [
        _md(head),
        _code('!pip -q install cleanlab numpy scipy joblib\nfrom google.colab import drive\ndrive.mount("/content/drive")\n'
              f'import os, json\nDRIVE = "{DRIVE_ROOT}"\nprint(sorted(os.listdir(DRIVE)))'),
        _embed(src_dir, "crc.py"),
        _embed(src_dir, "detectors.py"),
        _md("## Data, with labels this time"),
        _code(DATA_CELL.replace('"is_ok": cols["error_type"][i] == "ok",',
                                '"is_ok": cols["error_type"][i] == "ok",\n            "error_type": cols["error_type"][i],')
              .replace("ITEMS = []", "ITEMS_FULL = []").replace("ITEMS.append", "ITEMS_FULL.append")
              .replace("assert len(ITEMS) == 5700, len(ITEMS)", "ITEMS = ITEMS_FULL\nassert len(ITEMS) == 5700, len(ITEMS)")),
        _md("## The 600-item original-order sample (PREREG 9a)\n\nWrite this to Drive **before** the judge notebooks run, or rerun their 9a rows afterwards."),
        _code('import detectors\nkey_errors = [i for i, it in enumerate(ITEMS_FULL)\n'
              '              if it["error_type"] in {"wrong_groundtruth", "no_correct_answer", "multiple_correct_answers"}]\n'
              'sample = detectors.original_order_sample(ITEMS_FULL, key_errors, 600)\n'
              'with open(os.path.join(DRIVE, "original_order_sample.json"), "w") as fh:\n'
              '    json.dump(sample, fh)\n'
              'print(len(sample), "items;", len(key_errors), "key errors included")'),
        _md("## Join truth"),
        _code(ANALYSIS_CELL),
        _md("## Injection replicates: the main calibration test"),
        _code(REPLICATE_CELL),
        _md("""## GO / NO-GO

Check all four conditions from `PREREG.md` section 8 and write `RESULTS.md`:

- (a) coverage over the replicates >= 85%
- (b) the pooled Redux interval covers 181
- (c) the union misses >= 20% of the 181
- (d) |estimate - truth| <= 0.5 |union - truth|, on Redux and averaged over replicates

Report every detector's precision and recall, the full 31-cell capture pattern table, every estimator
with its goodness of fit, the estimated number of errors no detector found with its interval, and the
audit-cost reading. On NO-GO, name which condition failed."""),
    ]
    return _notebook(cells, gpu=False)


PREP_HEAD = """
# Prep: build the 600-item original-order sample

Runtime: **CPU** (no GPU, no High-RAM needed). Takes about a minute.

**Run this FIRST, before the judge notebooks.** It writes `original_order_sample.json` to Drive; the
judge notebooks read it and include those 600 rows in their pass. If you skip it, the judges simply omit
the section 9a rows and that exploratory check has to be run separately later.

It reads `error_type` only to make sure all 181 key errors are in the sample, as `PREREG.md` section 9a
requires. It never sees detector output -- none exists yet.
"""


def prep_notebook(src_dir: str) -> dict:
    cells = [
        _md(PREP_HEAD),
        _code('!pip -q install pyarrow numpy\nfrom google.colab import drive\ndrive.mount("/content/drive")\n'
              f'import os, json\nDRIVE = "{DRIVE_ROOT}"\nos.makedirs(DRIVE, exist_ok=True)'),
        _embed(src_dir, "detectors.py"),
        _code(DATA_CELL.replace('"is_ok": cols["error_type"][i] == "ok",',
                                '"is_ok": cols["error_type"][i] == "ok",\n            "error_type": cols["error_type"][i],')),
        _code('import detectors\n'
              'PRIMARY = {"wrong_groundtruth", "no_correct_answer", "multiple_correct_answers"}\n'
              'key_errors = [i for i, it in enumerate(ITEMS) if it["error_type"] in PRIMARY]\n'
              'assert len(key_errors) == 181, len(key_errors)\n'
              'sample = detectors.original_order_sample(ITEMS, key_errors, 600)\n'
              'assert len(sample) == 600 and set(key_errors) <= set(sample)\n'
              'with open(os.path.join(DRIVE, "original_order_sample.json"), "w") as fh:\n'
              '    json.dump(sample, fh)\n'
              'print("wrote", len(sample), "indices;", len(key_errors), "key errors included")'),
        _code(FINISH_CELL),
    ]
    return _notebook(cells, gpu=False)


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--src", default=here)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(here), "notebooks"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    written = []
    for model, short, note in JUDGES:
        path = os.path.join(args.out, f"judge_{short}.ipynb")
        with open(path, "w") as fh:
            json.dump(judge_notebook(args.src, model, short, note), fh, indent=1)
        written.append(path)
    path = os.path.join(args.out, "modern_arm.ipynb")
    with open(path, "w") as fh:
        json.dump(modern_notebook(args.src), fh, indent=1)
    written.append(path)
    path = os.path.join(args.out, "prep_sample.ipynb")
    with open(path, "w") as fh:
        json.dump(prep_notebook(args.src), fh, indent=1)
    written.append(path)
    path = os.path.join(args.out, "analysis.ipynb")
    with open(path, "w") as fh:
        json.dump(analysis_notebook(args.src), fh, indent=1)
    written.append(path)

    for p in written:
        print(f"wrote {p} ({os.path.getsize(p) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
