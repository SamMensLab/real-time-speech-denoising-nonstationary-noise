from pathlib import Path
import soundfile as sf
import numpy as np

def read_wav(path: Path, target_sr: int):
    x, sr = sf.read(str(path))
    if sr != target_sr:
        raise ValueError(f"Sample-rate mismatch for {path.name}: {sr} != {target_sr}")
    if x.ndim > 1:
        # downmix to mono
        x = np.mean(x, axis=1)
    return x.astype("float32"), sr

def write_wav(path: Path, x, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)
