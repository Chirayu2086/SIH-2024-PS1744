import numpy as np

from app.services.preprocess import FEATURE_NAMES


def explain_pixel(features: np.ndarray, x: int, y: int, top_k: int = 5):
    c, h, w = features.shape

    x = max(0, min(int(x), w - 1))
    y = max(0, min(int(y), h - 1))

    values = features[:, y, x].astype(float)

    indexed = [
        {"feature": FEATURE_NAMES[i], "value": float(values[i])}
        for i in range(len(values))
    ]

    indexed.sort(key=lambda item: abs(item["value"]), reverse=True)

    return {
        "x": x,
        "y": y,
        "top_features": indexed[:top_k],
        "note": "This explanation shows local input feature strength, not final causal attribution.",
    }
