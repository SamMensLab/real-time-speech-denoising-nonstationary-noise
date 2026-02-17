import time
import numpy as np
from .stft import stft, istft
from .mmse import wiener_mmse_gain
from .kalman import ARKalmanMagnitude
from .fusion import variance_optimal_fusion
from .metrics import real_time_factor

def enhance_ubke(x: np.ndarray, cfg: dict):
    """Enhance waveform x using UBKE. Returns (enhanced, rtf, mean_alpha)."""
    a = cfg["audio"]
    u = cfg["ubke"]
    kcfg = cfg.get("kalman", {})
    sr = int(a["sample_rate"])

    X = stft(
        x,
        n_fft=int(a["n_fft"]),
        win_length=int(a["win_length"]),
        hop_length=int(a["hop_length"]),
        window=str(a.get("window", "hann")),
    )
    mag = np.abs(X).astype(np.float32)
    phase = np.angle(X).astype(np.float32)

    init_frames = int(u["noise_update"]["init_frames"])
    init_frames = max(1, min(init_frames, mag.shape[0]))
    noise_psd = (np.mean(mag[:init_frames], axis=0).astype(np.float32) ** 2)

    noise_alpha = float(u["noise_update"]["alpha"])
    eps = float(u["epsilon"])

    tracker = ARKalmanMagnitude(
        p=int(u["ar_order_p"]),
        process_var=float(kcfg.get("process_var", 1e-3)),
        meas_var=float(kcfg.get("meas_var", 1e-2)),
    )

    enhanced_frames = []
    alphas = []

    t0 = time.perf_counter()

    for t in range(mag.shape[0]):
        # Spectral MMSE/Wiener branch
        G = wiener_mmse_gain(mag[t], noise_psd, eps=1e-10)
        mmse_mag = G * mag[t]

        # Temporal Kalman branch on magnitude
        kal_mag = tracker.step(mmse_mag)

        # Variance-optimal fusion
        fused_mag, alpha = variance_optimal_fusion(mmse_mag, kal_mag, eps=eps)
        alphas.append(alpha)

        Yt = fused_mag * np.exp(1j * phase[t])
        enhanced_frames.append(Yt)

        # Update noise PSD (simple minimum-statistics-like smoothing)
        noise_psd = noise_alpha * noise_psd + (1.0 - noise_alpha) * (mag[t] ** 2)

    Y = np.stack(enhanced_frames, axis=0)
    y = istft(
        Y,
        win_length=int(a["win_length"]),
        hop_length=int(a["hop_length"]),
        window=str(a.get("window", "hann")),
    )

    t1 = time.perf_counter()
    audio_seconds = len(x) / sr
    rtf = real_time_factor(t1 - t0, audio_seconds)

    return y.astype(np.float32), float(rtf), float(np.mean(alphas))
