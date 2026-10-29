"""verify_freeze.py: check that the files hashed in every test freeze record are still unchanged.

Run from the repository root:

  python scripts/verify_freeze.py            all records under reports/
  python scripts/verify_freeze.py --quiet    only mismatches and the summary

Every freeze record (reports/**/*freeze*.json) lists, under "inputs_sha256", the SHA-256 of each input file at the
moment the test split was opened (data set, code, checkpoints, stored forecasts). This script recomputes the hashes
and prints MATCH, CHANGED or MISSING per file. CHANGED means the file differs from the version used for that test
run: the test result is then no longer reproducible from the current file (for example src/control/battery_mpc.py
must stay byte-identical for notebooks 26 and 27). Exit code 1 if any file is CHANGED or MISSING.
Large files are hashed in 8 MiB blocks, so memory use stays small.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path, block=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(block):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    records = sorted(p for p in (ROOT / "reports").rglob("*freeze*.json"))
    if not records:
        sys.exit("no freeze records found under reports/")
    cache, bad, total = {}, 0, 0
    for rec in records:
        d = json.loads(rec.read_text())
        inputs = d.get("inputs_sha256", {})
        print(f"\n{rec.relative_to(ROOT)}  (frozen {d.get('frozen_at', '?')}, {len(inputs)} files)")
        for rel, want in inputs.items():
            total += 1
            p = ROOT / rel
            if not p.exists():
                status = "MISSING"
            else:
                if rel not in cache:
                    cache[rel] = sha256(p)
                status = "MATCH" if cache[rel] == want else "CHANGED"
            if status != "MATCH":
                bad += 1
            if status != "MATCH" or not args.quiet:
                print(f"  {status:<8} {rel}")
    print(f"\n{len(records)} records, {total} hashed inputs, {bad} changed or missing")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
