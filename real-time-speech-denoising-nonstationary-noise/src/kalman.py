import numpy as np

class ARKalmanMagnitude:
    """Lightweight AR(p) Kalman tracker operating on per-bin magnitudes.

    This is a causal, low-latency temporal smoother consistent with a state-space
    magnitude tracking view (used as the temporal branch of UBKE).
    """

    def __init__(self, p: int = 2, process_var: float = 1e-3, meas_var: float = 1e-2):
        self.p = int(p)
        self.Q = float(process_var)
        self.R = float(meas_var)
        self.state = None  # shape (F,)
        self.P = None      # shape (F,)

    def reset(self):
        self.state = None
        self.P = None

    def _init(self, F: int, init_val: np.ndarray):
        self.state = init_val.astype(np.float32).copy()
        self.P = np.ones(F, dtype=np.float32)

    def step(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        F = obs.shape[0]
        if self.state is None:
            self._init(F, obs)

        # Prediction: AR(1)-like (identity) for low compute
        pred = self.state

        # Scalar Kalman update per bin
        K = self.P / (self.P + self.R)
        est = pred + K * (obs - pred)

        self.P = (1.0 - K) * self.P + self.Q
        self.state = est
        return est
