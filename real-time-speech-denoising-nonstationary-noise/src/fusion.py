import numpy as np

def variance_optimal_fusion(mmse_mag: np.ndarray, kalman_mag: np.ndarray, eps: float = 1e-6):
    """Variance-optimal convex fusion with ε-regularization.

    alpha = var(kalman) / (var(mmse) + var(kalman) + eps)
    fused = alpha * mmse + (1-alpha) * kalman
    """
    mmse_mag = np.asarray(mmse_mag, dtype=np.float32)
    kalman_mag = np.asarray(kalman_mag, dtype=np.float32)

    v_mmse = float(np.var(mmse_mag))
    v_kal = float(np.var(kalman_mag))

    alpha = v_kal / (v_mmse + v_kal + eps)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    fused = alpha * mmse_mag + (1.0 - alpha) * kalman_mag
    return fused, alpha
