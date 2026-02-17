import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
from src.io_utils import read_wav, write_wav
from src.ubke import enhance_ubke

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--rtf_csv", default="results/rtf.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    sr = int(cfg["audio"]["sample_rate"])

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    wavs = sorted(inp.glob("*.wav"))
    if not wavs:
        raise FileNotFoundError(f"No wav files found in {inp}")

    rtf_rows = ["file,rtf,mean_alpha"]
    for wav in tqdm(wavs, desc="Enhancing"):
        x, _ = read_wav(wav, sr)
        y, rtf, mean_alpha = enhance_ubke(x, cfg)
        write_wav(out / wav.name, y, sr)
        rtf_rows.append(f"{wav.name},{rtf:.6f},{mean_alpha:.6f}")

    Path(args.rtf_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rtf_csv).write_text("\n".join(rtf_rows))
    print(f"Saved RTF summary to {args.rtf_csv}")

if __name__ == "__main__":
    main()
