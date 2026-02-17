import numpy as np

def wiener_mmse_gain(mag: np.ndarray, noise_psd: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """MMSE-style Wiener gain: G = xi / (xi + 1), with xi = |X|^2 / (Phi_n + eps)."""
    mag2 = mag ** 2
    xi = mag2 / (noise_psd + eps)
    G = xi / (xi + 1.0)
    return np.clip(G, 0.0, 1.0)
