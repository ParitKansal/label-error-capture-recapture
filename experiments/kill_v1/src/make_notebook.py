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
# (id, short name, note, smoke-check budget in session hours).
# The four budgets sum to 4.5, which is the stop in PREREG 10; Phi-3 is 14B and
# gets the larger share. A notebook that projects past its own share stops.
JUDGES = [
    ("meta-llama/Meta-Llama-3-8B-Instruct", "llama3", "gated: accept the licence and set HF_TOKEN", 1.0),
    ("mistralai/Mistral-7B-Instruct-v0.3", "mistral", "", 1.0),
    ("Qwen/Qwen2-7B-Instruct", "qwen2", "", 1.0),
    ("microsoft/Phi-3-medium-4k-instruct", "phi3", "14B, the slowest of the four", 1.5),
]

# PREREG section 9b, both verified on the Hub 2026-10-07.
# (id, short name, note, budget). Budgets sum to 1.8, the worst case in PREREG 10.
MODERN = [
    ("Qwen/Qwen3.8-27B-FP8", "qwen38", "30.9 GB, FP8 has no native Ampere support", 0.9),
    ("google/gemma-4-31B-it-qat-w4a16-ct", "gemma4", "23.3 GB, W4A16 runs natively on Ampere", 0.9),
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

# Two incompatible environments, one per notebook kind. They must never be mixed
# in one session.
#
# Judges (vLLM 0.11.0): Colab now ships transformers 5, and vLLM 0.11.0 fails at
# LLM() with "TokenizersBackend has no attribute all_special_tokens_extended".
# 0.11.0 needs the transformers 4 tokenizer API, so it is pinned below 5.
#
# Modern arm (vLLM 0.31.0): Qwen3.8 and Gemma 4 are 2026 architectures that
# vLLM 0.11.0 does not know at all, so that arm needs a current vLLM -- which in
# turn requires transformers >= 5.10.4. Verified against v0.31.0's
# requirements/common.txt, registry.py and docs/models/supported_models.md.
JUDGE_PINS = [
    "vllm==0.11.0",
    "transformers>=4.56,<5",
    "huggingface_hub>=0.26,<1.0",
]
MODERN_PINS = [
    "vllm==0.31.0",
    "transformers>=5.10.4,<5.18.0",
    "huggingface_hub>=1.31.0",
]


def setup_cell(pins: list[str], marker: str) -> str:
    """Install pinned dependencies, then restart the runtime exactly once.

    The restart is required, not cosmetic: Colab has already imported its own
    transformers by the time this runs, so a new version only takes effect in a
    fresh process. The marker file lives in /content, which survives a runtime
    restart but not a new VM, so the install happens once per machine and the
    restart cannot loop.
    """
    quoted = ", ".join(f'"{p}"' for p in pins)
    return f'''
import os, subprocess, sys, time
T_SESSION = time.time()   # Colab bills session time, so the clock starts here

PINS = [{quoted}]
MARKER = "/content/{marker}"

if not os.path.exists(MARKER):
    print("installing pinned dependencies, this takes a few minutes...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PINS], check=True)
    with open(MARKER, "w") as fh:
        fh.write(" ".join(PINS))
    print("\\nDependencies installed. RESTARTING THE RUNTIME NOW -- this is expected,")
    print("not a crash. When it reconnects, just Run all again: the install is")
    print("cached and this cell will skip straight past it.")
    os.kill(os.getpid(), 9)
else:
    print("dependencies already installed in this VM:")
    for p in PINS:
        print("  ", p)

# Colab exposes secrets to the kernel only. vLLM loads weights in a spawned
# engine process, so a gated model (Llama) fails with "Engine core
# initialization failed" unless the token is in the environment it inherits.
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN exported for the vLLM engine process")
except Exception as exc:
    print("no HF_TOKEN secret (fine for ungated models):", type(exc).__name__)
'''


VERSION_CHECK = '''
# Fail loudly here rather than deep inside LLM().
import importlib.metadata as md
import torch

for pkg in ("vllm", "transformers", "huggingface_hub", "tokenizers", "torch"):
    try:
        print(f"  {pkg:17s} {md.version(pkg)}")
    except Exception:
        print(f"  {pkg:17s} NOT INSTALLED")

vllm_v = md.version("vllm")
tf_v = md.version("transformers")
assert vllm_v == EXPECT_VLLM, f"expected vllm {EXPECT_VLLM}, got {vllm_v}. Restart the runtime and Run all again."
assert tf_v.split(".")[0] == EXPECT_TRANSFORMERS_MAJOR, (
    f"expected transformers {EXPECT_TRANSFORMERS_MAJOR}.x, got {tf_v}. "
    "Delete the marker file in /content and Run all again.")

print("\\nGPU:", torch.cuda.get_device_name(0))
cap = torch.cuda.get_device_capability(0)
print("capability:", cap, "-- (8, 0) is Ampere/A100: no native FP8, vLLM uses the Marlin weight-only path")
print("memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
assert cap[0] >= 8, f"this run needs an A100 (capability 8.0); got {cap}. Stop and pick a different runtime."

from google.colab import drive
drive.mount("/content/drive")
DRIVE = "__DRIVE_ROOT__"
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
gate_done = False
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

    if not gate_done:
        gate_done = True
        # --- automatic smoke check, PREREG 10 and 12 ---
        print("\\n--- first chunk: 3 raw model outputs ---")
        for row, out in list(zip(chunk, outs))[:3]:
            raw = out.outputs[0].text
            print(f"  [{row.kind} {row.row_id}]")
            print("   raw:", repr(raw[:160]))
            print("   parsed:", D.parse_verdict(raw))
        verdicts = [D.parse_verdict(o.outputs[0].text) for o in outs]
        from collections import Counter
        print("  verdict counts:", dict(Counter(verdicts)))
        res = D.smoke_check(verdicts,
                            session_elapsed_s=time.time() - T_SESSION,
                            generate_elapsed_s=time.time() - t0,
                            n_done=done_n, n_total=len(todo),
                            budget_hours=BUDGET_HOURS, label=SHORT)
        print("\\n" + res.message + "\\n", flush=True)
        if not res.ok:
            raise SystemExit(f"STOPPED by the smoke check: {'; '.join(res.reasons)}")

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


def judge_notebook(src_dir: str, model: str, short: str, note: str, budget: float) -> dict:
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
        _md("## Setup\n\nThe first run installs pinned dependencies and then **restarts the runtime on purpose**. "
            "When it reconnects, Run all again: the install is cached and skipped."),
        _code(setup_cell(JUDGE_PINS, ".deps_judge")),
        _code('EXPECT_VLLM = "0.11.0"\nEXPECT_TRANSFORMERS_MAJOR = "4"\n' + VERSION_CHECK.replace('__DRIVE_ROOT__', DRIVE_ROOT)),
        _code(f'MODEL = "{model}"\nSHORT = "{short}"\nBUDGET_HOURS = {budget}\n'
              f'print(MODEL, "| smoke-check budget", BUDGET_HOURS, "session hours")'),
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

for MODEL, SHORT, NOTE, BUDGET in MODERN_MODELS:
    T_MODEL = time.time()
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
    CHUNK = 512
    gate_done = False
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

        if not gate_done:
            gate_done = True
            print("\\n--- first chunk: 3 raw model outputs ---")
            for row, out in list(zip(chunk, outs))[:3]:
                raw = out.outputs[0].text
                print(f"  [{row.row_id}]")
                print("   raw:", repr(raw[:160]))
                print("   parsed:", D.parse_verdict(raw))
            verdicts = [D.parse_verdict(o.outputs[0].text) for o in outs]
            from collections import Counter
            print("  verdict counts:", dict(Counter(verdicts)))
            res = D.smoke_check(verdicts,
                                session_elapsed_s=time.time() - T_MODEL,
                                generate_elapsed_s=time.time() - t0,
                                n_done=n, n_total=len(todo),
                                budget_hours=BUDGET, label=SHORT)
            print("\\n" + res.message + "\\n", flush=True)
            if not res.ok:
                # exploratory arm: record it, skip this model, keep the other one
                with open(os.path.join(DRIVE, "modern_arm_failures.txt"), "a") as fh:
                    fh.write(f"{MODEL}\\tsmoke_check\\t{'; '.join(res.reasons)}\\n")
                print(f"SKIPPING {MODEL} and moving on (PREREG 9b).")
                break

        rate = n / (time.time() - t0)
        print(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo)-n)/max(rate,1e-9)/60:.1f} min", flush=True)
    del llm, tok
    gc.collect(); torch.cuda.empty_cache()

print("MODERN ARM DONE")
'''


def modern_notebook(src_dir: str) -> dict:
    models = ",\n    ".join(f'("{m}", "{s}", "{n}", {b})' for m, s, n, b in MODERN)
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
        _md("## Setup\n\nThis arm needs a **different environment** from the judge notebooks: vLLM 0.31.0 and "
            "transformers 5. Never run it in a session that has run a judge notebook. The first run installs "
            "and then **restarts the runtime on purpose**; Run all again when it reconnects."),
        _code(setup_cell(MODERN_PINS, ".deps_modern")),
        _code('EXPECT_VLLM = "0.31.0"\nEXPECT_TRANSFORMERS_MAJOR = "5"\n' + VERSION_CHECK.replace('__DRIVE_ROOT__', DRIVE_ROOT)),
        _code(f"MODERN_MODELS = [\n    {models},\n]\nfor m in MODERN_MODELS: print(m[0], '| budget', m[3], 'h')"),
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

LOAD_CELL = """
# Load the four judges and their option probabilities (PREREG 3). Labels are NOT used here.
import numpy as np, collections
JUDGE_SHORTS = ["llama3", "mistral", "qwen2", "phi3"]          # D1..D4, PREREG 3
DETECTOR_NAMES = ("D1_llama3", "D2_mistral", "D3_qwen2", "D4_phi3", "D5_cleanlab")
N = len(ITEMS)
ANS = np.array([int(it["answer"]) for it in ITEMS])

def _read_jsonl(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue            # a half-written last line after a disconnect
            out[r.get("row_id", r.get("item_index"))] = r   # resume can repeat a row: last wins
    return out

REDUX_FLAG = np.zeros((N, 4), dtype=np.int8)   # flag on the item's real key, shuffled order
REDUX_VERDICT = {}
INJ_FLAG = [dict() for _ in JUDGE_SHORTS]      # (item, wrong key) -> 0/1
ORIG_VERDICT = [dict() for _ in JUDGE_SHORTS]  # 9a original-order rows
UNPARSED = {}
for j, m in enumerate(JUDGE_SHORTS):
    rows = _read_jsonl(os.path.join(DRIVE, f"judge_{m}.jsonl")).values()
    kinds = collections.Counter(r["kind"] for r in rows)
    redux = {r["item_index"]: r["verdict"] for r in rows if r["kind"] == "redux"}
    assert len(redux) == N, f"{m}: {len(redux)} redux rows, expected {N} -- judge not finished"
    REDUX_VERDICT[m] = redux
    for i, v in redux.items():
        REDUX_FLAG[i, j] = int(v == "INCORRECT")
    for r in rows:
        if r["kind"] == "inject":
            INJ_FLAG[j][(r["item_index"], r["claimed_key_original"])] = int(r["verdict"] == "INCORRECT")
        elif r["kind"] == "original":
            ORIG_VERDICT[j][r["item_index"]] = r["verdict"]
    UNPARSED[m] = {k: sum(r["verdict"] == "unparsed" for r in rows if r["kind"] == k) / max(c, 1)
                   for k, c in kinds.items()}
    print(f"{m}: rows {dict(kinds)}  unparsed {', '.join(f'{k} {v:.1%}' for k, v in UNPARSED[m].items())}")
    if max(UNPARSED[m].values()) > 0.05:
        print(f"  WARNING: {m} unparsed rate above 5% -- reported as broken (PREREG 3, 12)")

# D5: average the four models' option probabilities (already mapped to original option order)
P = np.zeros((N, 4))
for m in JUDGE_SHORTS:
    pr = _read_jsonl(os.path.join(DRIVE, f"probs_{m}.jsonl"))
    assert len(pr) == N, f"{m}: probability pass has {len(pr)} of {N} items"
    for i, r in pr.items():
        P[int(i)] += np.asarray(r["probs"], dtype=float)
P /= len(JUDGE_SHORTS)
P = P / P.sum(axis=1, keepdims=True)

def d5_flags(labels):
    from cleanlab.filter import find_label_issues
    return find_label_issues(labels=np.asarray(labels), pred_probs=P,
                             filter_by="prune_by_noise_rate", n_jobs=1).astype(np.int8)

def build_flag_matrix(flipped_key):
    # (N, 5) flags for a benchmark whose keys are ANS except the flipped items (PREREG 7.4)
    flags = np.zeros((N, 5), dtype=np.int8)
    flags[:, :4] = REDUX_FLAG
    labels = ANS.copy()
    for i, k in flipped_key.items():
        labels[i] = k
        for j in range(4):
            flags[i, j] = INJ_FLAG[j][(i, k)]
    flags[:, 4] = d5_flags(labels)
    return flags
print("loaded; inject rows per judge:", [len(d) for d in INJ_FLAG])
"""

REDUX_CELL = """
# The Redux arm: realism test against the 181 expert-labelled key errors (PREREG 2, 4, 5, 8).
import crc
MASTER_SEED = 20261006
is_key = np.array([t["is_key_error"] for t in truth])
is_expert = np.array([t["is_expert"] for t in truth])
is_ok = np.array([t["is_ok"] for t in truth])
FLAGS_REDUX = build_flag_matrix({})

print("per detector on Redux (precision excludes the 32 expert items):")
for j, name in enumerate(DETECTOR_NAMES):
    f = FLAGS_REDUX[:, j].astype(bool)
    tp = int((f & is_key).sum()); fp = int((f & ~is_key & ~is_expert).sum())
    print(f"  {name:12s} flags {int(f.sum()):5d}  captures {tp:4d}  recall {tp/181:6.1%}  "
          f"precision {tp/max(tp+fp,1):6.1%}  flags on ok items {int((f & is_ok).sum()):5d}")

data_redux = crc.CaptureData.from_flags(FLAGS_REDUX, is_key, DETECTOR_NAMES)
pat, cnt = data_redux.cell_counts()
print("\\ncapture-pattern table (D1..D5), 31 cells:")
for p_, c_ in zip(pat, cnt):
    print("  ", "".join(map(str, p_)), int(c_))

REDUX_EST = crc.summary(data_redux, n_boot=2000, seed=MASTER_SEED)
PRIMARY_REDUX = REDUX_EST[0]
print(f"\\nunion found {data_redux.n_found} of 181 key errors")
for e in REDUX_EST:
    gof = e.extra.get("gof_p")
    print(f"  {e.name:22s} n_hat {e.n_hat:8.1f}  CI [{e.lo:7.1f}, {e.hi:7.1f}]  ok={e.ok}  "
          + (f"gof_p={gof:.3f}  " if gof is not None and gof == gof else "") + e.note)
print(f"\\nestimated key errors NO detector found: {PRIMARY_REDUX.n_hat - data_redux.n_found:.1f} "
      f"[{PRIMARY_REDUX.lo - data_redux.n_found:.1f}, {PRIMARY_REDUX.hi - data_redux.n_found:.1f}]")
union_flags = int(FLAGS_REDUX.any(axis=1).sum())
print(f"audit cost: a human checks {union_flags} flagged items to find {data_redux.n_found} key errors")

print("\\nsecondary, by supercategory (not part of GO):")
cats = np.array([t["supercategory"] for t in truth])
for c in sorted(set(cats)):
    m_ = cats == c
    d_ = crc.CaptureData.from_flags(FLAGS_REDUX[m_], is_key[m_], DETECTOR_NAMES)
    e_ = crc.loglinear(d_, n_boot=2000, seed=MASTER_SEED)
    print(f"  {c:34s} truth {int(is_key[m_].sum()):3d}  union {d_.n_found:3d}  "
          f"n_hat {e_.n_hat:7.1f}  CI [{e_.lo:6.1f}, {e_.hi:6.1f}]  covers {e_.covers(int(is_key[m_].sum()))}")

print("\\nexploratory 9a: share of the 181 key errors each judge calls CORRECT (endorses the wrong key)")
for j, m in enumerate(JUDGE_SHORTS):
    both = [i for i in ORIG_VERDICT[j] if is_key[i]]
    if both:
        o = np.mean([ORIG_VERDICT[j][i] == "CORRECT" for i in both])
        s_ = np.mean([REDUX_VERDICT[m][i] == "CORRECT" for i in both])
        print(f"  {m:8s} original order {o:6.1%}   shuffled {s_:6.1%}   (n={len(both)})")
"""

GO_CELL = """
# GO / NO-GO, exactly PREREG 8. Nothing here is tunable.
truth_r = 181
est_r, union_r = PRIMARY_REDUX.n_hat, data_redux.n_found
mean_abs_est = float(np.mean([abs(r["n_hat"] - r["truth"]) for r in results]))
mean_abs_union = float(np.mean([abs(r["union"] - r["truth"]) for r in results]))
cond = {
    "a": coverage >= 0.85,
    "b": PRIMARY_REDUX.covers(truth_r),
    "c": (truth_r - union_r) / truth_r >= 0.20,
    "d": abs(est_r - truth_r) <= 0.5 * abs(union_r - truth_r) and mean_abs_est <= 0.5 * mean_abs_union,
}
detail = {
    "a": f"coverage {coverage:.1%} over {len(results)} replicates (need >= 85%)",
    "b": f"Redux CI [{PRIMARY_REDUX.lo:.1f}, {PRIMARY_REDUX.hi:.1f}] vs 181",
    "c": f"union finds {union_r}/181, misses {(truth_r - union_r) / truth_r:.1%} (need >= 20%)",
    "d": f"Redux |est-181| {abs(est_r - truth_r):.1f} vs half |union-181| {0.5 * abs(union_r - truth_r):.1f}; "
         f"replicates mean |est-266| {mean_abs_est:.1f} vs half mean |union-266| {0.5 * mean_abs_union:.1f}",
}
for k in "abcd":
    print(f"({k}) {'PASS' if cond[k] else 'FAIL'}  {detail[k]}")
if union_r >= 0.9 * truth_r or not cond["a"] or (not cond["b"] and not cond["d"]):
    verdict = "NO-GO"
elif all(cond.values()):
    verdict = "GO"
else:
    verdict = "PARTIAL (failed: " + ", ".join(k for k in "abcd" if not cond[k]) + ")"
print("\\nreplicates: mean union", np.mean([r["union"] for r in results]),
      "| unstable share", np.mean([not r["ok"] for r in results]),
      "| median gof p", np.nanmedian([r["gof_p"] for r in results]))
print(f"\\nPREREG.md blob {PREREG_COMMIT}")
print(f"VERDICT: {verdict}")
with open(os.path.join(DRIVE, "analysis_summary.json"), "w") as fh:
    json.dump({"verdict": verdict, "conditions": {k: bool(v) for k, v in cond.items()}, "detail": detail,
               "prereg_blob": PREREG_COMMIT,
               "redux": [{"name": e.name, "n_hat": e.n_hat, "lo": e.lo, "hi": e.hi, "ok": e.ok, "note": e.note}
                         for e in REDUX_EST],
               "replicates": [{k: (v if not isinstance(v, (list, tuple)) else str(v)) for k, v in r.items()}
                              for r in results]}, fh, indent=1, default=float)
print("saved analysis_summary.json to Drive")
"""

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
              '_p = os.path.join(DRIVE, "original_order_sample.json")\n'
              'if os.path.exists(_p):\n'
              '    assert sorted(json.load(open(_p))) == sorted(sample), "9a sample on Drive differs"\n'
              'else:\n'
              '    json.dump(sample, open(_p, "w"))\n'
              'print(len(sample), "items;", len(key_errors), "key errors included")'),
        _md("## Join truth"),
        _code(ANALYSIS_CELL),
        _md("## Load detector outputs and build D5"),
        _code(LOAD_CELL),
        _md("## Redux arm: the realism test"),
        _code(REDUX_CELL),
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
        _code(GO_CELL),
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
    for model, short, note, budget in JUDGES:
        path = os.path.join(args.out, f"judge_{short}.ipynb")
        with open(path, "w") as fh:
            json.dump(judge_notebook(args.src, model, short, note, budget), fh, indent=1)
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
