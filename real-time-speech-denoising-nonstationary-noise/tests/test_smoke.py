import yaml
import numpy as np
from src.ubke import enhance_ubke

def test_smoke_runs():
    cfg = yaml.safe_load(open("configs/ubke_default.yaml", "r"))
    sr = cfg["audio"]["sample_rate"]
    x = np.random.randn(sr).astype(np.float32) * 0.01
    y, rtf, a = enhance_ubke(x, cfg)
    assert y.shape[0] > 0
    assert rtf > 0
