from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from app.core.config import settings
from app.models.sarangnet import SARangNet


_MODEL = None


def get_model() -> SARangNet:
    global _MODEL

    if _MODEL is None:
        model = SARangNet(
            in_channels=settings.input_channels,
            out_channels=settings.output_channels,
        )

        if settings.model_path.exists():
            state_dict = torch.load(settings.model_path, map_location=settings.device)
            model.load_state_dict(state_dict)
        else:
            # Random weights for hackathon demo if no checkpoint exists.
            # In final submission, always use trained weights.
            pass

        model.to(settings.device)
        _MODEL = model

    return _MODEL


def feather_weight(h: int, w: int, overlap: int) -> np.ndarray:
    overlap = max(1, min(overlap, h // 4, w // 4))

    wy = np.ones(h, dtype=np.float32)
    wx = np.ones(w, dtype=np.float32)

    ramp = np.linspace(0.05, 1.0, overlap, dtype=np.float32)

    wy[:overlap] = ramp
    wy[-overlap:] = ramp[::-1]

    wx[:overlap] = ramp
    wx[-overlap:] = ramp[::-1]

    weight = np.outer(wy, wx)
    return weight.astype(np.float32)


def pad_to_multiple(img: np.ndarray, tile_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    _, h, w = img.shape

    pad_h = (tile_size - h % tile_size) % tile_size
    pad_w = (tile_size - w % tile_size) % tile_size

    padded = np.pad(
        img,
        ((0, 0), (0, pad_h), (0, pad_w)),
        mode="reflect",
    )

    return padded, (pad_h, pad_w)


def crop_to_original(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return img[:, :h, :w]


def run_tiled_inference(
    features: np.ndarray,
    tile_size: int = 256,
    overlap: int = 32,
    uncertainty: bool = False,
    n_samples: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        rgb: float32 array (H, W, 3), range 0-1
        confidence: float32 array (H, W), range 0-1
    """
    model = get_model()
    device = torch.device(settings.device)

    c, h, w = features.shape

    padded, _ = pad_to_multiple(features, tile_size)
    _, ph, pw = padded.shape

    stride = max(1, tile_size - overlap)
    weight = feather_weight(tile_size, tile_size, overlap)

    output_channels = settings.output_channels

    predictions = []

    for _ in range(n_samples if uncertainty else 1):
        if uncertainty:
            model.train()
        else:
            model.eval()

        out_sum = np.zeros((output_channels, ph, pw), dtype=np.float32)
        weight_sum = np.zeros((1, ph, pw), dtype=np.float32)

        with torch.no_grad():
            for y in range(0, ph - tile_size + 1, stride):
                for x in range(0, pw - tile_size + 1, stride):
                    tile = padded[:, y:y + tile_size, x:x + tile_size]
                    tile_tensor = torch.from_numpy(tile).unsqueeze(0).float().to(device)

                    pred = model(tile_tensor)
                    pred_np = pred.squeeze(0).cpu().numpy()

                    out_sum[:, y:y + tile_size, x:x + tile_size] += pred_np * weight
                    weight_sum[:, y:y + tile_size, x:x + tile_size] += weight

        weight_sum = np.maximum(weight_sum, 1e-6)
        stitched = out_sum / weight_sum
        stitched = crop_to_original(stitched, h, w)
        predictions.append(stitched)

    predictions = np.stack(predictions, axis=0)

    mean_pred = np.mean(predictions, axis=0)

    if uncertainty and len(predictions) > 1:
        std_pred = np.std(predictions, axis=0)
        rgb_std = std_pred[:3].mean(axis=0)
        confidence = 1.0 - np.clip(rgb_std / max(rgb_std.max(), 1e-6), 0.0, 1.0)
    else:
        if output_channels >= 4:
            confidence = mean_pred[3]
        else:
            confidence = np.ones((h, w), dtype=np.float32)

    rgb = mean_pred[:3].transpose(1, 2, 0)
    rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)

    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)

    return rgb, confidence
