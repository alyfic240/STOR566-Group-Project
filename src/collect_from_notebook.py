"""Unpack the result files that the v5 "remaining runs" notebook printed as base64 blocks.

The notebook prints, for every result file,
    @@FILE <path relative to the results folder> <size in bytes> <sha256>
    <base64 lines>
    @@END <path>
Colab stores cell output inside the .ipynb, so the notebook file itself carries the results.

Usage:  python3 -I collect_from_notebook.py <notebook.ipynb> <results_dir> [--overwrite]
Every block is checked against its size and SHA-256 before it is written. Existing files are left alone
unless --overwrite is given (identical content is reported as "same").
"""
import base64
import hashlib
import json
import os
import sys


def blocks(nb):
    for ci, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        text = ""
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream" and o.get("name", "stdout") == "stdout":
                t = o.get("text", "")
                text += t if isinstance(t, str) else "".join(t)
        # blocks may also sit in a cell's source (used when results are carried in a helper notebook)
        src = cell.get("source", "")
        src = src if isinstance(src, str) else "".join(src)
        if src.startswith("@@FILE "):
            text += ("\n" if text and not text.endswith("\n") else "") + src
        lines = text.split("\n")
        i = 0
        while i < len(lines):
            if lines[i].startswith("@@FILE "):
                _, rel, size, sha = lines[i].split(" ")
                j = i + 1
                body = []
                while j < len(lines) and not lines[j].startswith("@@END "):
                    body.append(lines[j].strip())
                    j += 1
                complete = j < len(lines) and lines[j] == f"@@END {rel}"
                yield ci, rel, int(size), sha, "".join(body), complete
                i = j
            i += 1


def main(nb_path, out_dir, overwrite=False):
    nb = json.load(open(nb_path, encoding="utf-8"))
    n_ok = n_bad = 0
    for ci, rel, size, sha, b64, complete in blocks(nb):
        if os.path.isabs(rel) or ".." in rel.split("/"):
            print(f"REFUSED  {rel}: path leaves the results folder")
            n_bad += 1
            continue
        try:
            data = base64.b64decode(b64, validate=True) if complete else None
        except Exception as e:
            data = None
            print(f"BAD      {rel}: base64 error {e}")
        if data is None or len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            print(f"BAD      {rel} (cell {ci}): incomplete or damaged block "
                  f"(complete={complete}, got {0 if data is None else len(data)} of {size} bytes)")
            n_bad += 1
            continue
        dst = os.path.join(out_dir, rel)
        if os.path.exists(dst):
            same = open(dst, "rb").read() == data
            if same or not overwrite:
                print(f"{'same    ' if same else 'KEPT OLD'} {rel}")
                n_ok += same
                continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)
        print(f"written  {rel} ({size} bytes, sha256 {sha[:12]})")
        n_ok += 1
    print(f"{n_ok} file(s) verified, {n_bad} problem(s)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args[0], args[1], overwrite="--overwrite" in sys.argv))
