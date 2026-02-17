import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.metrics_csv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # summary statistics
    numeric_cols = [c for c in df.columns if c != "file"]
    summary = df[numeric_cols].describe().transpose()
    summary.to_csv(out / "summary_stats.csv")

    # mean row
    means = df[numeric_cols].mean(numeric_only=True)
    means.to_csv(out / "means.csv", header=["mean"])

    print(f"Wrote tables to {out}")

if __name__ == "__main__":
    main()
