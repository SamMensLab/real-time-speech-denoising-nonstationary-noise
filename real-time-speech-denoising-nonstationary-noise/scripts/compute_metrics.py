import argparse
from pathlib import Path
import csv
import yaml
import numpy as np

from src.io_utils import read_wav
from src.metrics import lsd_framewise

def try_import_pesq():
    try:
        from pesq import pesq
        return pesq
    except Exception:
        return None

def try_import_stoi():
    try:
        from pystoi.stoi import stoi
        return stoi
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_config", required=True)
    ap.add_argument("--clean_dir", required=True)
    ap.add_argument("--enhanced_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.eval_config, "r"))
    sr = int(cfg["audio"]["sample_rate"])
    mcfg = cfg["metrics"]

    compute_pesq = bool(mcfg.get("compute_pesq", False))
    compute_stoi = bool(mcfg.get("compute_stoi", True))
    compute_lsd = bool(mcfg.get("compute_lsd", True))

    pesq_fn = try_import_pesq() if compute_pesq else None
    stoi_fn = try_import_stoi() if compute_stoi else None

    clean_dir = Path(args.clean_dir)
    enh_dir = Path(args.enhanced_dir)

    clean_files = sorted(clean_dir.glob("*.wav"))
    if not clean_files:
        raise FileNotFoundError(f"No wav files found in {clean_dir}")

    rows = []
    for cfile in clean_files:
        efile = enh_dir / cfile.name
        if not efile.exists():
            continue

        c, _ = read_wav(cfile, sr)
        e, _ = read_wav(efile, sr)
        n = min(len(c), len(e))
        c = c[:n]
        e = e[:n]

        row = {"file": cfile.name}

        if compute_lsd:
            row["lsd"] = lsd_framewise(c, e)

        if compute_stoi and stoi_fn is not None:
            row["stoi"] = float(stoi_fn(c, e, sr, extended=False))
        elif compute_stoi:
            row["stoi"] = ""

        if compute_pesq and pesq_fn is not None:
            mode = str(mcfg.get("pesq_mode", "wb"))
            # PESQ expects 8k (nb) or 16k (wb)
            row["pesq"] = float(pesq_fn(sr, c, e, mode))
        elif compute_pesq:
            row["pesq"] = ""

        rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["file"]
    if compute_lsd: fieldnames.append("lsd")
    if compute_stoi: fieldnames.append("stoi")
    if compute_pesq: fieldnames.append("pesq")

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote metrics to {out}")
    if compute_pesq and pesq_fn is None:
        print("Note: PESQ library not available. Install `pesq` or set compute_pesq: false.")

if __name__ == "__main__":
    main()
