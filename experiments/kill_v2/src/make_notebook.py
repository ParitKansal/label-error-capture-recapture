"""Write the Colab notebooks for the kill experiment.

Each notebook embeds the src files it needs via %%writefile, so a notebook is
self-contained once uploaded: no git clone, no Drive copy of the source.

    python make_notebook.py --out ../notebooks

Produces:
  judge_<model>.ipynb   x3  one per model family. GPU (A100). Runs that model's
                            16,360 judge prompts, its option-probability pass
                            for confident learning, and its share of the
                            600-item original-order check.
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

DRIVE_ROOT = "/content/drive/MyDrive/label_error_crc/kill_v2"
REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"

# PREREG section 3. (id, short name, bf16 weights, note)
# (id, short name, note, smoke-check budget in session hours). PREREG 3 and 7.
# All three are 2026 models; none is gated. Budgets sum to 3.8, inside the
# 4.1-hour expectation and well inside the 7-hour stop.
JUDGES = [
    ("google/gemma-4-31B-it", "gemma4",
     "62.5 GB bf16; fallback google/gemma-4-31B-it-qat-w4a16-ct if it will not fit", 1.2),
    ("Qwen/Qwen3.8-27B-FP8", "qwen38",
     "thinking OFF via enable_thinking=False; v1 lost this model to thinking", 1.0),
    ("mistralai/Mistral-Small-4-119B-2603-NVFP4", "mistral4",
     "Mistral-native format, no config.json; vLLM maps it to MistralLarge3ForCausalLM", 1.6),
]

# Every v2 model is a 2026 architecture, so every notebook uses the same current
# environment. vLLM 0.31.0 verified against its registry.py and
# docs/models/supported_models.md for Gemma4ForConditionalGeneration,
# Qwen3_5ForConditionalGeneration and MistralLarge3ForCausalLM, and its
# requirements/common.txt for the transformers range.
JUDGE_PINS = [
    "vllm==0.31.0",
    "transformers>=5.10.4,<5.18.0",
    "huggingface_hub>=1.31.0",
    "mistral-common>=1.5.0",
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
# v2 has no original-order arm (that was v1 section 9a, already answered).
SAMPLE = set()
ROWS = D.build_work_list(ITEMS, SAMPLE)
from collections import Counter
print(Counter(r.kind for r in ROWS), "total", len(ROWS))
'''

RESUME_CELL = '''
# Resume: read back whatever this model already finished.
OUT = os.path.join(DRIVE, f"judge_{SHORT}.jsonl")
assert "kill_v2" in OUT, "v2 must never write into the v1 Drive folder"
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

FIRST_TOKEN_CELL = '''
# PREREG 3: before the real pass, check the FIRST generated token is a verdict.
# v1 lost its Qwen3.8 arm to a thinking model that spent every output token
# reasoning. This costs 8 prompts and catches it in seconds.
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
llm = LLM(model=MODEL, gpu_memory_utilization=0.90, max_model_len=4096,
          **EXTRA_LLM_KWARGS)
CHAT_KWARGS = D.thinking_kwargs(MODEL)
print("chat template kwargs (thinking off):", CHAT_KWARGS)

def build_prompts(rows):
    prompts, metas = [], []
    for row in rows:
        text, meta = D.render_row(row, ITEMS[row.item_index])
        chat = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True, **CHAT_KWARGS)
        prompts.append(chat + D.VERDICT_PREFILL)   # first generated token = the verdict
        metas.append(meta)
    return prompts, metas

def token_logprobs(out):
    lp = out.outputs[0].logprobs
    if not lp:
        return {}
    return {info.decoded_token: info.logprob for info in lp[0].values()}

PARAMS = SamplingParams(temperature=0.0, max_tokens=4, logprobs=20)
probe_rows = [r for r in ROWS if r.kind == D.REDUX][:8]
pp, _ = build_prompts(probe_rows)
probe = llm.generate(pp, PARAMS)
good = 0
for out in probe:
    tl = token_logprobs(out)
    ok = D.first_token_is_verdict(tl)
    good += ok
    print(f"  first-token ok={ok}  text={out.outputs[0].text[:40]!r}  "
          f"top={sorted(tl, key=tl.get, reverse=True)[:4]}")
if good < 7:
    raise SystemExit(
        f"STOP: only {good}/8 first tokens were a verdict word for {MODEL}.\\n"
        "This model is reasoning before answering, or the prefill did not take.\\n"
        "Do not run the full pass; report this (PREREG 3).")
print(f"first-token check PASSED ({good}/8)")
'''

JUDGE_RUN_CELL = '''
CHUNK = 512
t0 = time.time()
gate_done = False
for start in range(0, len(todo), CHUNK):
    chunk = todo[start:start + CHUNK]
    prompts, metas = build_prompts(chunk)
    outs = llm.generate(prompts, PARAMS)
    with open(OUT, "a") as fh:
        for row, meta, out in zip(chunk, metas, outs):
            tl = token_logprobs(out)
            score = D.incorrect_probability(tl)       # PREREG 4, the primary signal
            text = D.VERDICT_PREFILL + out.outputs[0].text
            fh.write(json.dumps({
                "row_id": row.row_id, "kind": row.kind, "item_index": row.item_index,
                "claimed_key_original": row.claimed_key_original,
                "perm": meta["perm"], "claimed_letter": meta["claimed_letter"],
                "raw": text[:120],
                "score": score,                        # None when unscored
                "verdict": D.parse_verdict(text),      # secondary, reported only
            }) + "\\n")
    done_n = start + len(chunk)

    if not gate_done:
        gate_done = True
        print("\\n--- first chunk: 3 raw model outputs ---")
        for row, out in list(zip(chunk, outs))[:3]:
            raw = D.VERDICT_PREFILL + out.outputs[0].text
            print(f"  [{row.kind} {row.row_id}]")
            print("   raw:", repr(raw[:120]))
            print("   parsed:", D.parse_verdict(raw),
                  " score:", D.incorrect_probability(token_logprobs(out)))
        verdicts = [D.parse_verdict(D.VERDICT_PREFILL + o.outputs[0].text) for o in outs]
        scores = [D.incorrect_probability(token_logprobs(o)) for o in outs]
        unscored = sum(x is None for x in scores) / max(len(scores), 1)
        from collections import Counter
        print("  verdict counts (secondary, reported only):", dict(Counter(verdicts)))
        print(f"  unscored rate: {unscored:.1%}  (PREREG 4: unscored items are never flagged)")
        # v2's primary signal is the score (PREREG 4), so the smoke check's 5% limit
        # applies to UNSCORED items; the verdict text is secondary and only reported.
        res = D.smoke_check(["CORRECT" if x is not None else "unparsed" for x in scores],
                            session_elapsed_s=time.time() - T_SESSION,
                            generate_elapsed_s=time.time() - t0,
                            n_done=done_n, n_total=len(todo),
                            budget_hours=BUDGET_HOURS, label=SHORT)
        print("\\n" + res.message + "\\n", flush=True)
        if not res.ok or unscored > 0.05:
            raise SystemExit(
                f"STOPPED by the smoke check: {'; '.join(res.reasons) or f'unscored rate {unscored:.1%}'}")

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
            tokenize=False, add_generation_prompt=True, **CHAT_KWARGS))
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
        _code('EXPECT_VLLM = "0.31.0"\nEXPECT_TRANSFORMERS_MAJOR = "5"\n' + VERSION_CHECK.replace('__DRIVE_ROOT__', DRIVE_ROOT)),
        _code(f'MODEL = "{model}"\nSHORT = "{short}"\nBUDGET_HOURS = {budget}\n'
              '# Mistral-Small-4 ships Mistral-native format only (no config.json), so vLLM\n'
              '# needs the mistral tokenizer/config/load modes. PREREG 3.\n'
              'EXTRA_LLM_KWARGS = ({"tokenizer_mode": "mistral", "config_format": "mistral",\n'
              '                     "load_format": "mistral"} if "mistral" in MODEL.lower() else {})\n'
              f'print(MODEL, "| smoke-check budget", BUDGET_HOURS, "session hours")\n'
              'print("extra LLM kwargs:", EXTRA_LLM_KWARGS)'),
        _embed(src_dir, "detectors.py"),
        _code(DATA_CELL),
        _code(WORKLIST_CELL),
        _code(RESUME_CELL),
        _md("## First-token check\n\n8 prompts. Stops the notebook if this model reasons instead of answering."),
        _code(FIRST_TOKEN_CELL),
        _md("## Judge pass\n\n16,360 prompts: 5,700 Redux + 10,660 injection scoring. Each row stores P(INCORRECT) as well as the verdict."),
        _code(JUDGE_RUN_CELL),
        _md("## Probability pass for confident learning (D5)\n\nSame session, weights already loaded."),
        _code(PROBS_CELL),
        _md("## Finish\n\nFlush Drive, then **close the session**."),
        _code(FINISH_CELL),
    ]
    return _notebook(cells, gpu=True)


ANALYSIS_CELL = '''
# Join truth to detector output. THIS IS THE ONLY PLACE EXPERT LABELS ARE READ.
# Record the commit hash of PREREG.md first (PREREG 9, 13).
# Blob hash of experiments/kill_v2/PREREG.md as committed. Filled in by
# make_notebook.py so the author never needs to run git in Colab (PREREG 9).
PREREG_COMMIT = "d983111828355c30e2ab7bb5884a6b371624ff53"
assert PREREG_COMMIT, "record the pre-registration blob hash before running the analysis"
print("pre-registration blob:", PREREG_COMMIT)

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
import detectors as D
# Load the three judges' SCORES and their option probabilities (PREREG 3, 4).
# Labels are NOT used in this cell.
import numpy as np, collections
JUDGE_SHORTS = ["gemma4", "qwen38", "mistral4"]                 # D1..D3
DETECTOR_NAMES = ("D1_gemma4", "D2_qwen38", "D3_mistral4", "D4_cleanlab")
N_DET = 4
BUDGET = D.FLAG_BUDGET                                           # 285, PREREG 4
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
            out[r.get("row_id", r.get("item_index"))] = r   # resume can repeat: last wins
    return out

REDUX_SCORE = [dict() for _ in JUDGE_SHORTS]   # item -> P(INCORRECT) on its real key
INJ_SCORE = [dict() for _ in JUDGE_SHORTS]     # (item, wrong key) -> P(INCORRECT)
REDUX_VERDICT = {}                             # secondary, reported only
UNSCORED = {}
for j, m in enumerate(JUDGE_SHORTS):
    rows = _read_jsonl(os.path.join(DRIVE, f"judge_{m}.jsonl")).values()
    kinds = collections.Counter(r["kind"] for r in rows)
    redux = {r["item_index"]: r for r in rows if r["kind"] == "redux"}
    assert len(redux) == N, f"{m}: {len(redux)} redux rows, expected {N} -- judge not finished"
    REDUX_SCORE[j] = {i: r["score"] for i, r in redux.items()}
    REDUX_VERDICT[m] = {i: r["verdict"] for i, r in redux.items()}
    for r in rows:
        if r["kind"] == "inject":
            INJ_SCORE[j][(r["item_index"], r["claimed_key_original"])] = r["score"]
    n_uns = sum(v is None for v in REDUX_SCORE[j].values())
    UNSCORED[m] = n_uns / N
    print(f"{m}: rows {dict(kinds)}  unscored on Redux {UNSCORED[m]:.2%}  "
          f"inject scores {len(INJ_SCORE[j])}")
    if UNSCORED[m] > 0.05:
        print(f"  WARNING: {m} unscored rate above 5% -- reported as broken (PREREG 4)")

# D4's source: average the three models' option probabilities (already in original
# option order). PREREG 3.
P = np.zeros((N, 4))
for m in JUDGE_SHORTS:
    pr = _read_jsonl(os.path.join(DRIVE, f"probs_{m}.jsonl"))
    assert len(pr) == N, f"{m}: probability pass has {len(pr)} of {N} items"
    for i, r in pr.items():
        P[int(i)] += np.asarray(r["probs"], dtype=float)
P /= len(JUDGE_SHORTS)
P = P / P.sum(axis=1, keepdims=True)

def build_flag_matrix(flipped_key):
    \"\"\"(N, 4) flags for a benchmark whose keys are ANS except the flipped items.

    THE BUDGET IS APPLIED PER BENCHMARK (PREREG 4): each judge flags exactly its
    top 285 items by P(INCORRECT), and D4 the 285 items whose claimed key the
    ensemble is least confident in. Flipped items take their injection score,
    every other item its Redux score.
    \"\"\"
    labels = ANS.copy()
    for i, k in flipped_key.items():
        labels[i] = k
    flags = np.zeros((N, N_DET), dtype=np.int8)
    for j in range(len(JUDGE_SHORTS)):
        scores = dict(REDUX_SCORE[j])
        for i, k in flipped_key.items():
            scores[i] = INJ_SCORE[j][(i, k)]
        for i in D.top_k_flags(scores, BUDGET):
            flags[i, j] = 1
    conf = {i: float(P[i, labels[i]]) for i in range(N)}
    for i in D.bottom_k_flags(conf, BUDGET):
        flags[i, 3] = 1
    assert flags.sum(axis=0).tolist() == [BUDGET] * N_DET, flags.sum(axis=0)
    return flags

print("loaded; inject scores per judge:", [len(d) for d in INJ_SCORE])
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
print("\\ncapture-pattern table (D1..D4), 15 cells:")
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

print("\\nSECONDARY, reported only, never used for GO (PREREG 4):")
print("what the natural INCORRECT verdict would have flagged, had there been no budget:")
for j, m in enumerate(JUDGE_SHORTS):
    nat = np.array([REDUX_VERDICT[m].get(i) == "INCORRECT" for i in range(N)])
    tp = int((nat & is_key).sum()); fp = int((nat & ~is_key & ~is_expert).sum())
    print(f"  {m:9s} would flag {int(nat.sum()):5d}  captures {tp:4d}  recall {tp/181:6.1%}  "
          f"precision {tp/max(tp+fp,1):6.1%}")
nat_union = np.zeros(N, dtype=bool)
for m in JUDGE_SHORTS:
    nat_union |= np.array([REDUX_VERDICT[m].get(i) == "INCORRECT" for i in range(N)])
print(f"  verdict-union would flag {int(nat_union.sum())} items and catch "
      f"{int((nat_union & is_key).sum())} of 181 -- compare with the budgeted union above.")
print("  (v1 failed here: its verdict-union flagged 4,145 of 5,700 and caught 94.5%.)")
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
import detectors            # used inside one_replicate, which runs in a worker
import detectors as D       # build_flag_matrix closes over this name

MASTER_SEED = 20261006
N_REPLICATES = 200
N_BOOT = 2000          # PREREG 9; the local end-to-end test overrides both
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
    flags = build_flag_matrix(flipped_key)          # (5700, 4) 0/1, budget applied per replicate
    is_error = np.zeros(len(ITEMS), dtype=bool)
    is_error[sorted(injected)] = True
    data = crc.CaptureData.from_flags(flags, is_error, DETECTOR_NAMES)
    # PRIMARY: nonparametric bootstrap (PREREG 5a). crc.loglinear defaults to it.
    est = crc.loglinear(data, n_boot=N_BOOT, seed=MASTER_SEED)
    alt = crc.loglinear(data, n_boot=N_BOOT, seed=MASTER_SEED, ci_method="parametric")
    return {
        "rep": rep, "truth": len(injected), "union": data.n_found,
        "n_hat": est.n_hat, "lo": est.lo, "hi": est.hi, "ok": est.ok,
        "gof_p": est.extra["gof_p"], "terms": est.extra["terms"],
        "covers": est.covers(len(injected)),
        "covers_parametric": alt.covers(len(injected)),   # secondary, reported only
    }

import os as _os
print("CPU cores available:", _os.cpu_count(),
      "-- this cell is CPU-bound; pick the runtime with the most cores.")
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(one_replicate)(r) for r in range(N_REPLICATES))
coverage = np.mean([r["covers"] for r in results])
cov_par = np.mean([r["covers_parametric"] for r in results])
print(f"empirical coverage over {N_REPLICATES} replicates: {coverage:.1%}  (GO needs >= 85%)")
print(f"  secondary, parametric interval: {cov_par:.1%}  (reported only, PREREG 5a)")
'''


def analysis_notebook(src_dir: str) -> dict:
    head = """
# Analysis

Runtime: **CPU, High-RAM**. No GPU. Run this only after the judge notebooks have finished and
`PREREG.md` is committed.

This is the **only** notebook that reads `error_type`. It:
1. records the commit hash of `PREREG.md`;
2. joins truth to detector output;
3. applies the 285-item flagging budget to every detector (PREREG 4);
4. builds D4 by confident learning over the stored option probabilities;
5. runs the estimators on the Redux arm;
6. runs the 200+ injection replicates and reports empirical coverage;
7. checks each of the four GO conditions.

The primary interval is the **nonparametric** bootstrap (PREREG 5a), a change from v1; the parametric one
is reported alongside it as a secondary.

Nothing in it may be retuned after looking at results. Anything not named in `PREREG.md` goes in the
exploratory section.
"""
    cells = [
        _md(head),
        _code('!nproc\n!pip -q install cleanlab numpy scipy joblib pyarrow\n'
              'import os\nprint("CPU cores:", os.cpu_count(),\n'
              '      "-- the replicate cell is CPU-bound; use the runtime with the most cores.")\n'
              'from google.colab import drive\ndrive.mount("/content/drive")\n'
              f'import json\nDRIVE = "{DRIVE_ROOT}"\nprint(sorted(os.listdir(DRIVE)))'),
        _embed(src_dir, "crc.py"),
        _embed(src_dir, "detectors.py"),
        _md("## Data, with labels this time"),
        _code(DATA_CELL.replace('"is_ok": cols["error_type"][i] == "ok",',
                                '"is_ok": cols["error_type"][i] == "ok",\n            "error_type": cols["error_type"][i],')
              .replace("ITEMS = []", "ITEMS_FULL = []").replace("ITEMS.append", "ITEMS_FULL.append")
              .replace("assert len(ITEMS) == 5700, len(ITEMS)", "ITEMS = ITEMS_FULL\nassert len(ITEMS) == 5700, len(ITEMS)")),
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
    path = os.path.join(args.out, "analysis.ipynb")
    with open(path, "w") as fh:
        json.dump(analysis_notebook(args.src), fh, indent=1)
    written.append(path)

    for p in written:
        print(f"wrote {p} ({os.path.getsize(p) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
