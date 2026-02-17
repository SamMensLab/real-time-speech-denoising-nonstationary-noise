# Real-Time Speech Denoising under Non-Stationary Noise (UBKE) — Reproducibility Package

Repository URL: https://github.com/SamMensLab/real-time-speech-denoising-nonstationary-noise

This repository contains the **source code**, **configuration files**, and **evaluation scripts** needed to reproduce the experiments reported in the accepted manuscript on real-time speech denoising under non-stationary noise.

## What is included
- **UBKE implementation** (spectral MMSE-style enhancement + AR(p) Kalman tracking + variance-optimal fusion)
- Centralized **YAML configs** mirroring manuscript parameters (16 kHz, 32 ms frames, 50% overlap, FFT=512, AR order p=2, ε=1e-6)
- Scripts to:
  - Enhance noisy audio (`run_enhancement.py`)
  - Compute metrics (`compute_metrics.py`) including STOI, LSD, and optional PESQ (if installed)
  - Generate summary tables (`make_tables.py`)
  - Run a self-contained demo (`quick_demo.py`) without external datasets

> Note: This repo does not redistribute copyrighted datasets. See `data/README_DATA.md` for data layout.

---

## 1) Installation

### Python (recommended)
Tested with:
- Python 3.10+
- NumPy 1.26.4
- SciPy 1.11.4

Install:
```bash
pip install -r requirements.txt
```

Optional: conda environment
```bash
conda env create -f environment.yml
conda activate ubke-repro
```

---

## 2) Quick demo (no dataset required)
```bash
python scripts/quick_demo.py --config configs/ubke_default.yaml --out_dir results/demo
```

---

## 3) Reproducing evaluation (requires your dataset folders)

### Expected folder structure
```
data/
  test_noisy/    # noisy wavs
  test_clean/    # clean reference wavs (same filenames as noisy)
```

### A) Enhance audio
```bash
python scripts/run_enhancement.py \
  --config configs/ubke_default.yaml \
  --input_dir data/test_noisy \
  --output_dir results/enhanced
```

### B) Compute metrics (STOI + LSD; PESQ optional)
```bash
python scripts/compute_metrics.py \
  --eval_config configs/eval_default.yaml \
  --clean_dir data/test_clean \
  --enhanced_dir results/enhanced \
  --out_csv results/metrics.csv
```

### C) Generate summary tables
```bash
python scripts/make_tables.py \
  --metrics_csv results/metrics.csv \
  --out_dir results/tables
```

---

## 4) Reproducibility notes
- Randomness is controlled via a fixed seed where applicable.
- Runtime performance is reported as **RTF (real-time factor)**: processing_time / audio_duration.
- Software and system information can be recorded in `SYSTEM_INFO.md` and `MATLAB_VERSION.txt`.

---

## 5) Citation
See `CITATION.cff`.

Accessed on 17 February 2026.
