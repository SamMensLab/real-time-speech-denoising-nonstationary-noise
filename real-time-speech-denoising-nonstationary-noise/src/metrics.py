import time
import numpy as np

def real_time_factor(proc_seconds: float, audio_seconds: float) -> float:
    return float(proc_seconds / max(audio_seconds, 1e-9))

def lsd_framewise(clean: np.ndarray, est: np.ndarray, n_fft: int = 512, hop: int = 256, win: int = 512, eps: float = 1e-8) -> float:
    """Framewise log-spectral distance (LSD) on magnitude spectra."""
    clean = np.asarray(clean, dtype=np.float32)
    est = np.asarray(est, dtype=np.float32)
    n = min(len(clean), len(est))
    clean = clean[:n]
    est = est[:n]

    def frame_stft(x):
        if len(x) < win:
            x = np.pad(x, (0, win - len(x)))
        T = 1 + (len(x) - win) // hop
        out = []
        w = np.hanning(win).astype(np.float32)
        for i in range(T):
            seg = x[i*hop:i*hop+win] * w
            out.append(np.abs(np.fft.rfft(seg, n_fft)))
        return np.stack(out, axis=0)

    C = frame_stft(clean)
    E = frame_stft(est)
    lc = np.log(C + eps)
    le = np.log(E + eps)
    return float(np.mean(np.sqrt(np.mean((lc - le)**2, axis=1))))
