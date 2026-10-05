"""Recompute every number in DATA.md from the pinned MMLU-Redux 2.0 revision.

Downloads the 57 Arrow files at the pinned revision and the official MMLU
subject -> supercategory map, then prints the counts. No Hugging Face
`datasets` dependency; only pyarrow and the standard library.

    pip install pyarrow
    python dataset_facts.py --cache-dir ./_mmlu_redux_cache
"""

import argparse
import collections
import json
import os
import urllib.parse
import urllib.request

DATASET = "edinburgh-dawg/mmlu-redux-2.0"
REVISION = "372ea425445d51e1ba1188c56e5e893f8138621f"
CATEGORIES_URL = "https://raw.githubusercontent.com/hendrycks/test/master/categories.py"

PRIMARY = ("wrong_groundtruth", "no_correct_answer", "multiple_correct_answers")
ALL_TYPES = PRIMARY + ("bad_question_clarity", "bad_options_clarity", "expert", "ok")


def download(cache_dir):
    """Fetch every .arrow file at the pinned revision. Returns [(subject, path)]."""
    tree_url = (
        f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}?recursive=true"
    )
    with urllib.request.urlopen(tree_url, timeout=60) as fh:
        tree = json.load(fh)
    paths = [e["path"] for e in tree if e["type"] == "file" and e["path"].endswith(".arrow")]
    if len(paths) != 57:
        raise SystemExit(f"expected 57 arrow files at {REVISION}, found {len(paths)}")
    os.makedirs(cache_dir, exist_ok=True)
    out = []
    for path in sorted(paths):
        subject = path.split("/")[0]
        local = os.path.join(cache_dir, f"{subject}.arrow")
        if not os.path.exists(local):
            url = (
                f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/"
                + urllib.parse.quote(path)
            )
            urllib.request.urlretrieve(url, local)
        out.append((subject, local))
    return out


def supercategory_map():
    """subject -> one of the 4 MMLU supercategories, from the official categories.py."""
    with urllib.request.urlopen(CATEGORIES_URL, timeout=60) as fh:
        src = fh.read().decode()
    ns = {}
    exec(compile(src, "categories.py", "exec"), ns)  # the file is two plain dict literals
    out = {}
    for subject, subcats in ns["subcategories"].items():
        for cat, members in ns["categories"].items():
            if subcats[0] in members:
                out[subject] = cat
    return out


def load_rows(files):
    import pyarrow as pa
    import pyarrow.ipc as ipc

    rows = []
    for subject, path in files:
        with pa.memory_map(path, "rb") as src:
            table = ipc.open_stream(src).read_all()
        cols = table.to_pydict()
        for i in range(table.num_rows):
            rows.append(
                {
                    "subject": subject,
                    "question": cols["question"][i],
                    "choices": tuple(cols["choices"][i]),
                    "answer": cols["answer"][i],
                    "error_type": cols["error_type"][i],
                    "source": cols["source"][i],
                    "correct_answer": cols["correct_answer"][i],
                }
            )
    return rows


def nonempty(value):
    return value is not None and str(value).strip() not in ("", "nan", "None")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="./_mmlu_redux_cache")
    args = ap.parse_args()

    files = download(args.cache_dir)
    sub2cat = supercategory_map()
    rows = load_rows(files)

    print(f"revision {REVISION}")
    print(f"items {len(rows)}  subjects {len({r['subject'] for r in rows})}")
    print(f"all items have 4 choices: {all(len(r['choices']) == 4 for r in rows)}")
    print(f"all answers in 0..3: {all(0 <= r['answer'] <= 3 for r in rows)}")
    print(f"items with 4 distinct choices: {sum(len(set(r['choices'])) == 4 for r in rows)}")

    counts = collections.Counter(r["error_type"] for r in rows)
    print("\nerror_type counts")
    for key in ALL_TYPES:
        print(f"  {key:26s} {counts[key]:5d}")
    primary = sum(counts[k] for k in PRIMARY)
    print(f"  {'PRIMARY (key errors)':26s} {primary:5d}")
    print(f"  {'SECONDARY (any non-ok)':26s} {len(rows) - counts['ok']:5d}")

    print("\nby supercategory")
    for cat in sorted({sub2cat[r['subject']] for r in rows}):
        sub = [r for r in rows if sub2cat[r["subject"]] == cat]
        c = collections.Counter(r["error_type"] for r in sub)
        print(
            f"  {cat:34s} n={len(sub):5d} ok={c['ok']:5d} "
            f"primary={sum(c[k] for k in PRIMARY):4d} nonok={len(sub) - c['ok']:4d}"
        )

    print("\ndata-quality checks")
    for key in ALL_TYPES:
        sub = [r for r in rows if r["error_type"] == key]
        if sub:
            print(
                f"  correct_answer non-empty for {key:26s} "
                f"{sum(nonempty(r['correct_answer']) for r in sub):4d} / {len(sub)}"
            )
    print(f"  source non-empty: {sum(nonempty(r['source']) for r in rows)} / {len(rows)}")
    qcount = collections.Counter(r["question"] for r in rows)
    print(f"  duplicated question texts: {sum(1 for v in qcount.values() if v > 1)}")
    print(f"  extra rows from duplicates: {sum(v - 1 for v in qcount.values() if v > 1)}")
    exact = collections.Counter((r["question"], r["choices"], r["answer"]) for r in rows)
    print(f"  exact duplicate rows beyond first: {sum(v - 1 for v in exact.values() if v > 1)}")

    per_subject = collections.Counter(
        r["subject"] for r in rows if r["error_type"] in PRIMARY
    )
    zero = [s for s in {r["subject"] for r in rows} if per_subject[s] == 0]
    print(f"\nsubjects with zero key errors: {len(zero)}")
    print("top subjects by key errors: " + ", ".join(f"{s} {n}" for s, n in per_subject.most_common(8)))


if __name__ == "__main__":
    main()
