import argparse
from pathlib import Path
import numpy as np
import soundfile as sf
import yaml
from src.ubke import enhance_ubke

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="results/demo")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    sr = int(cfg["audio"]["sample_rate"])

    # Synthetic clean-like signal + non-stationary noise envelope
    t = np.linspace(0, 3.0, int(sr * 3.0), endpoint=False)
    clean = 0.1*np.sin(2*np.pi*220*t) + 0.05*np.sin(2*np.pi*440*t)
    env = 0.2 + 0.8*(np.sin(2*np.pi*0.7*t)**2)
    noise = 0.06*np.random.randn(len(t))*env
    noisy = (clean + noise).astype(np.float32)

    y, rtf, mean_alpha = enhance_ubke(noisy, cfg)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sf.write(out/"noisy.wav", noisy, sr)
    sf.write(out/"enhanced.wav", y, sr)

    print(f"Demo complete. RTF={rtf:.3f}, mean alpha={mean_alpha:.3f}")
    print(f"Outputs written to: {out}")

if __name__ == "__main__":
    main()
