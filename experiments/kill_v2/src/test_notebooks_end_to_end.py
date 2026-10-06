"""Run the analysis notebook end to end, locally, on synthetic detector output.

Why this exists: v1 shipped an analysis notebook with an undefined function in
it (`build_flag_matrix`), which was only discovered on Colab after the GPU work
was already paid for. This harness executes every cell of the generated
notebook in order, in one namespace, so any name used before it is defined
fails here instead of there.

What is real and what is fake:
  real  -- the pinned MMLU-Redux dataset, crc.py, detectors.py, and every cell
           of analysis.ipynb exactly as generated.
  fake  -- the judge JSONL and probability JSONL, fabricated from
           detectors.build_work_list in the real on-disk format, with scores
           drawn so that key errors score a little higher than ok items.

Run:  python -u test_notebooks_end_to_end.py --out /tmp/e2e.log
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

THIS = os.path.dirname(os.path.abspath(__file__))


def fabricate(drive: str, items: list[dict], shorts: list[str], seed: int = 7) -> None:
    """Write judge_<m>.jsonl and probs_<m>.jsonl in the real format."""
    import numpy as np

    sys.path.insert(0, THIS)
    import detectors as D

    rng = np.random.default_rng(seed)
    rows = D.build_work_list(items, set())
    key_error = np.array([it["error_type"] in
                          {"wrong_groundtruth", "no_correct_answer", "multiple_correct_answers"}
                          for it in items])

    for j, short in enumerate(shorts):
        path = os.path.join(drive, f"judge_{short}.jsonl")
        with open(path, "w") as fh:
            for row in rows:
                # injected (wrong-key) rows and true key errors score higher, with
                # heavy overlap between judges, which is the realistic hard case
                base = 0.25 + 0.10 * j
                if row.kind == D.INJECT:
                    base += 0.35
                elif key_error[row.item_index]:
                    base += 0.20
                score = float(np.clip(rng.normal(base, 0.18), 0.001, 0.999))
                if rng.random() < 0.004:
                    score = None            # unscored, as a real run will have some
                verdict = "unparsed" if score is None else (
                    "INCORRECT" if score > 0.5 else "CORRECT")
                fh.write(json.dumps({
                    "row_id": row.row_id, "kind": row.kind, "item_index": row.item_index,
                    "claimed_key_original": row.claimed_key_original,
                    "perm": [int(x) for x in D.option_permutation(row.item_index)],
                    "claimed_letter": "A", "raw": f"VERDICT: {verdict}",
                    "score": score, "verdict": verdict,
                }) + "\n")

        ppath = os.path.join(drive, f"probs_{short}.jsonl")
        with open(ppath, "w") as fh:
            for i, it in enumerate(items):
                p = rng.dirichlet(np.ones(4) * 0.7)
                if not key_error[i]:
                    p[it["answer"]] += 1.2   # usually agrees with a good key
                p = p / p.sum()
                fh.write(json.dumps({"item_index": i, "probs": p.tolist()}) + "\n")
    print(f"fabricated {len(rows)} rows x {len(shorts)} judges in {drive}", flush=True)


def run_notebook(path: str, workdir: str, overrides: dict) -> dict:
    """Execute every code cell in order in one namespace. Returns that namespace."""
    nb = json.load(open(path))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns: dict = {"__name__": "__notebook__"}
    cwd = os.getcwd()
    os.chdir(workdir)
    try:
        for n, cell in enumerate(cells, 1):
            body = "".join(cell["source"])
            if body.startswith("%%writefile"):
                fname = body.splitlines()[0].split()[1]
                with open(os.path.join(workdir, fname), "w") as fh:
                    fh.write("\n".join(body.splitlines()[1:]))
                print(f"  cell {n:2d}: wrote {fname}", flush=True)
                continue
            # drop shell lines and the Colab-only drive mount
            lines = []
            for line in body.splitlines():
                st = line.lstrip()
                if st.startswith(("!", "%")):
                    continue
                if "google.colab" in line or "drive.mount" in line or "flush_and_unmount" in line:
                    continue
                lines.append(line)
            src = "\n".join(lines)
            for key, val in overrides.items():
                src = src.replace(key, val)
            t0 = time.time()
            try:
                exec(compile(src, f"{os.path.basename(path)}:cell{n}", "exec"), ns)
            except Exception as exc:
                print(f"  cell {n:2d}: FAILED after {time.time()-t0:.1f}s "
                      f"-- {type(exc).__name__}: {exc}", flush=True)
                raise
            print(f"  cell {n:2d}: ok ({time.time()-t0:.1f}s)", flush=True)
    finally:
        os.chdir(cwd)
    return ns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebooks", default=os.path.join(os.path.dirname(THIS), "notebooks"))
    ap.add_argument("--replicates", type=int, default=12)
    ap.add_argument("--n-boot", type=int, default=60)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    work = tempfile.mkdtemp(prefix="kill_v2_e2e_")
    drive = os.path.join(work, "drive")
    os.makedirs(drive, exist_ok=True)
    print(f"workdir {work}", flush=True)

    # the real dataset, loaded the same way the notebook does
    sys.path.insert(0, THIS)
    import dataset_facts as DF
    files = DF.download(os.path.join(work, "cache"))
    sub2cat = DF.supercategory_map()
    raw = DF.load_rows(files)
    items = [{"subject": r["subject"], "supercategory": sub2cat[r["subject"]],
              "question": r["question"], "choices": list(r["choices"]),
              "answer": int(r["answer"]), "error_type": r["error_type"],
              "is_ok": r["error_type"] == "ok"} for r in raw]
    assert len(items) == 5700, len(items)
    print(f"loaded {len(items)} real items", flush=True)

    fabricate(drive, items, ["gemma4", "qwen38", "mistral4"])

    overrides = {
        f'DRIVE = "/content/drive/MyDrive/label_error_crc/kill_v2"': f'DRIVE = {drive!r}',
        "N_REPLICATES = 200": f"N_REPLICATES = {args.replicates}",
        "N_BOOT = 2000": f"N_BOOT = {args.n_boot}",
        "n_boot=2000": f"n_boot={args.n_boot}",
        'CACHE = "/content/mmlu_redux"': f'CACHE = {os.path.join(work, "cache")!r}',
    }
    path = os.path.join(args.notebooks, "analysis.ipynb")
    print(f"\n=== {os.path.basename(path)} ===", flush=True)
    ns = run_notebook(path, work, overrides)

    verdict = None
    summary = os.path.join(drive, "analysis_summary.json")
    if os.path.exists(summary):
        verdict = json.load(open(summary))["verdict"]
    print(f"\nEND TO END OK. coverage={ns.get('coverage')} verdict={verdict}", flush=True)
    print("(the verdict is meaningless here -- the detector output is synthetic; "
          "what this proves is that every cell runs and every name is defined)", flush=True)
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":      # macOS spawns workers by re-importing this file
    raise SystemExit(main())
