"""reproduce.py: run the project notebooks in order and export their outputs (one command for Weeks 3-7).

Run from the repository root in the dcpower environment:

  python scripts/reproduce.py                      show the plan only; nothing runs
  python scripts/reproduce.py --run                run every validation-only notebook from 06 to 25
  python scripts/reproduce.py --run --stage quantum
  python scripts/reproduce.py --run --only 24 25
  python scripts/reproduce.py --run --from 17 --to 22b --keep-going
  python scripts/reproduce.py --run --skip-done    resume: skip notebooks already reproduced without error

What it does, per notebook:
  1. reads notebooks/<name>.ipynb (the file on disk is never changed unless --in-place is given);
  2. disables the notebook's own export cell (`%run ../src/tools/nb_outputs_to_md.py ...`) in the copy it runs;
  3. executes all cells with nbclient in notebooks/ (the working directory every notebook expects), using the
     kernel named in the notebook (dcpower) unless --kernel is given;
  4. saves the executed copy to notebooks/_reproduced/<name>.ipynb and its outputs to
     notebooks/_reproduced/<name>_outputs.md (with src/tools/nb_outputs_to_md.py);
  5. appends one row to reports/reproduce_log.csv (time, notebook, status, seconds, output).

Frozen test protocol. Notebooks 16, 23, 24, 26 and 27 open the test split (each writes a freeze record before its
single test run). They are SKIPPED unless --include-test is given; running them again writes a second freeze record
next to the first, which must then be reported as a re-run. Verify the existing records with
`python scripts/verify_freeze.py` instead.

Notebooks 00-05 (environment checks, raw-data download from S3, EDA of Weeks 1-2) are historical and never run by
this script: 02 downloads raw telemetry and uses the project's old location. The data pipeline is documented in
README.md (sections "Data pipeline") and docs/CODE_MAP.md.
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
OUT_DIR = NB_DIR / "_reproduced"
LOG_F = ROOT / "reports" / "reproduce_log.csv"
EXPORT_MARK = "nb_outputs_to_md.py"

# (id, file, stage, opens the test split, note)
NOTEBOOKS = [
    ("06", "06_model_dataset.ipynb", "data", False, "modelling dataset v1.1 (validity, split, ramp labels)"),
    ("07", "07_baselines.ipynb", "benchmark", False, "naive and linear baselines"),
    ("08", "08_classical_models.ipynb", "benchmark", False, "AutoARIMA, AutoETS, LightGBM with Optuna"),
    ("09", "09_week3_completion.ipynb", "benchmark", False, "LSTM / GRU, 3 seeds (long: network training)"),
    ("10", "10_week4_nf_smoke.ipynb", "benchmark", False, "smoke test of N-HiTS, PatchTST, DLinear, NLinear"),
    ("11", "11_week4_nf_models.ipynb", "benchmark", False, "neuralforecast architectures, 3 seeds (long)"),
    ("12", "12_week4_val_analysis.ipynb", "benchmark", False, "validation analysis, DM with Holm, ramp event"),
    ("13", "13_week4_chronos_smoke.ipynb", "benchmark", False, "Chronos-Bolt smoke test (downloads the model)"),
    ("14", "14_week4_chronos_val.ipynb", "benchmark", False, "Chronos-Bolt zero-shot on validation"),
    ("15", "15_week4_timesfm_val.ipynb", "benchmark", False, "TimesFM 2.5 zero-shot on validation"),
    ("16", "16_week4_test_evaluation.ipynb", "benchmark", True, "frozen test run of the benchmark"),
    ("17", "17_week5_qnn_smoke.ipynb", "quantum", False, "hybrid QNN checks, timing, gradient variance"),
    ("18", "18_week5_qnn_val.ipynb", "quantum", False, "hybrid QNN and controls, 3 seeds"),
    ("19", "19_week5_qrc.ipynb", "quantum", False, "quantum reservoir and controls"),
    ("20", "20_week5_completion.ipynb", "quantum", False, "Week 5 table and costs"),
    ("21", "21_week6_fraction_qubits.ipynb", "quantum", False, "data fraction and qubit sweep (about 3 h on the M1)"),
    ("22", "22_week6_noise.ipynb", "quantum", False, "inference noise (shots, Heron-like)"),
    ("22b", "22b_week6_quantum_kernel.ipynb", "quantum", False, "quantum kernel (about 21 min of fits)"),
    ("23", "23_week6_quantum_test.ipynb", "quantum", True, "frozen test run of the quantum models"),
    ("24", "24_week7_conformal.ipynb", "week7", True, "conformal intervals (validation screen, then test)"),
    ("25", "25_week7_shap.ipynb", "week7", False, "SHAP on the frozen LightGBM (validation only)"),
    ("26", "26_week7_mpc.ipynb", "week7", True, "battery MPC (validation sweep, then test)"),
    ("27", "27_week7_mpc_band.ipynb", "week7", True, "uncertainty-aware MPC (validation, then test)"),
]
ORDER = [n[0] for n in NOTEBOOKS]
STAGES = ("data", "benchmark", "quantum", "week7")
DONE = set()


def select(args):
    ids = ORDER
    if args.stage:
        ids = [n[0] for n in NOTEBOOKS if n[2] in args.stage]
    if args.only:
        unknown = [i for i in args.only if i not in ORDER]
        if unknown:
            sys.exit(f"unknown notebook id(s): {unknown}; known: {ORDER}")
        ids = [i for i in ids if i in args.only]
    if args.start:
        ids = [i for i in ids if ORDER.index(i) >= ORDER.index(args.start)]
    if args.end:
        ids = [i for i in ids if ORDER.index(i) <= ORDER.index(args.end)]
    return [n for n in NOTEBOOKS if n[0] in ids]


def done_ok():
    """Notebooks whose latest logged run succeeded and whose reproduced copy exists (for --skip-done)."""
    last = {}
    if LOG_F.exists():
        with LOG_F.open() as fh:
            for r in csv.DictReader(fh):
                last[r["notebook"]] = r["status"]
    return {f for f, s in last.items() if s == "ok" and (OUT_DIR / f).exists()}


def plan(rows, args):
    print(f"repository: {ROOT}")
    print(f"{'id':<4} {'notebook':<34} {'stage':<10} action")
    for nid, f, stage, test, note in rows:
        if not (NB_DIR / f).exists():
            act = "MISSING"
        elif test and not args.include_test:
            act = "skip (opens the test split; --include-test to run)"
        elif args.skip_done and f in DONE:
            act = "skip (already reproduced without error)"
        else:
            act = "run"
        print(f"{nid:<4} {f:<34} {stage:<10} {act}   # {note}")


def disable_export(nb):
    """Blank the notebook's own export cell in the copy that runs; returns the original sources for --in-place."""
    saved = {}
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and EXPORT_MARK in cell.source:
            saved[i] = cell.source
            cell.source = "# export cell disabled by scripts/reproduce.py (outputs are exported after the run)"
    return saved


def log_row(row):
    LOG_F.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG_F.exists()
    with LOG_F.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "notebook", "status", "seconds", "output", "detail"])
        w.writerow(row)


def run_one(f, args):
    import nbformat
    from nbclient import NotebookClient
    sys.path.insert(0, str(ROOT / "src" / "tools"))
    import nb_outputs_to_md as EXP

    src = NB_DIR / f
    nb = nbformat.read(src, as_version=4)
    saved = disable_export(nb)
    kernel = args.kernel or nb.metadata.get("kernelspec", {}).get("name", "python3")
    client = NotebookClient(nb, timeout=args.timeout, kernel_name=kernel,
                            resources={"metadata": {"path": str(NB_DIR)}}, allow_errors=False)
    t0 = time.time()
    status, detail = "ok", ""
    try:
        client.execute()
    except Exception as e:  # noqa: BLE001  (the partial notebook is still saved for inspection)
        name = getattr(e, "ename", type(e).__name__)
        value = getattr(e, "evalue", str(e).strip().splitlines()[-1] if str(e).strip() else "")
        status, detail = "error", f"{name}: {value}"[:300]
    secs = time.time() - t0
    if args.in_place:
        for i, s in saved.items():
            nb.cells[i].source = s
        dst = src
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        dst = OUT_DIR / f
    nbformat.write(nb, dst)
    md = EXP.export(dst, (OUT_DIR if not args.in_place else NB_DIR) / (Path(f).stem + "_outputs.md"))
    log_row([datetime.now().isoformat(timespec="seconds"), f, status, f"{secs:.0f}", str(md.relative_to(ROOT)), detail])
    print(f"  {status} in {secs / 60:.1f} min -> {dst.relative_to(ROOT)}" + (f" | {detail}" if detail else ""))
    return status == "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="execute (without it only the plan is printed)")
    ap.add_argument("--stage", nargs="+", choices=STAGES, help="restrict to one or more stages")
    ap.add_argument("--only", nargs="+", metavar="ID", help="notebook ids, e.g. 24 25 22b")
    ap.add_argument("--from", dest="start", choices=ORDER, help="first notebook id")
    ap.add_argument("--to", dest="end", choices=ORDER, help="last notebook id")
    ap.add_argument("--include-test", action="store_true",
                    help="also run notebooks that open the test split (writes a second freeze record)")
    ap.add_argument("--skip-done", action="store_true", help="skip notebooks whose last logged run succeeded")
    ap.add_argument("--keep-going", action="store_true", help="continue after a failing notebook")
    ap.add_argument("--in-place", action="store_true", help="save outputs into the original notebooks")
    ap.add_argument("--kernel", help="kernel name (default: the notebook's own, dcpower)")
    ap.add_argument("--timeout", type=int, default=None, help="per-cell timeout in seconds (default: none)")
    args = ap.parse_args()

    global DONE
    DONE = done_ok() if args.skip_done else set()
    rows = select(args)
    if not rows:
        sys.exit("nothing selected")
    plan(rows, args)
    if not args.run:
        print("\nplan only; add --run to execute")
        return
    if not (ROOT / "data" / "processed" / "model_dataset.parquet").exists() and rows[0][0] != "06":
        sys.exit("data/processed/model_dataset.parquet is missing: run notebook 06 first (--only 06)")
    if args.include_test:
        print("\nWARNING: test-split notebooks will run again and write second freeze records.")

    done, failed = 0, []
    for nid, f, stage, test, note in rows:
        if not (NB_DIR / f).exists() or (test and not args.include_test):
            continue
        if args.skip_done and f in DONE:
            continue
        print(f"\n[{datetime.now():%H:%M:%S}] {nid} {f}  ({note})")
        if run_one(f, args):
            done += 1
        else:
            failed.append(f)
            if not args.keep_going:
                break
    print(f"\nfinished: {done} ok, {len(failed)} failed" + (f": {failed}" if failed else "") + f"; log {LOG_F.relative_to(ROOT)}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
