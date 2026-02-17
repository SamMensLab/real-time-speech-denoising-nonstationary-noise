import numpy as np
from scipy.signal import get_window

def stft(x: np.ndarray, n_fft: int, win_length: int, hop_length: int, window: str = "hann") -> np.ndarray:
    """Short-time Fourier transform (real-input, rFFT). Returns shape (T, F)."""
    x = np.asarray(x, dtype=np.float32)
    win = get_window(window, win_length, fftbins=True).astype(np.float32)

    if len(x) < win_length:
        x = np.pad(x, (0, win_length - len(x)))

    n_frames = 1 + (len(x) - win_length) // hop_length
    frames = np.empty((n_frames, win_length), dtype=np.float32)

    for i in range(n_frames):
        start = i * hop_length
        frames[i] = x[start:start + win_length] * win

    X = np.fft.rfft(frames, n=n_fft, axis=1)
    return X

def istft(X: np.ndarray, win_length: int, hop_length: int, window: str = "hann") -> np.ndarray:
    """Inverse STFT for rFFT frames. Overlap-add with window-squared normalization."""
    X = np.asarray(X)
    T, F = X.shape
    n_fft = (F - 1) * 2
    win = get_window(window, win_length, fftbins=True).astype(np.float32)

    y_len = (T - 1) * hop_length + win_length
    y = np.zeros(y_len, dtype=np.float32)
    wsum = np.zeros(y_len, dtype=np.float32)

    frames = np.fft.irfft(X, n=n_fft, axis=1)[:, :win_length].astype(np.float32)
    for i in range(T):
        start = i * hop_length
        y[start:start + win_length] += frames[i] * win
        wsum[start:start + win_length] += win ** 2

    wsum[wsum < 1e-8] = 1.0
    y /= wsum
    return y
