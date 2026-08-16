from pathlib import Path
from typing import Dict

import numpy as np
import rasterio


def rgb_to_uint8(rgb: np.ndarray) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def write_rgb_geotiff(path: Path, rgb: np.ndarray, meta: Dict) -> None:
    out_meta = meta.copy()
    out_meta.update(
        count=3,
        dtype="uint8",
        compress="lzw",
    )

    rgb_uint8 = rgb_to_uint8(rgb)

    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(rgb_uint8[:, :, 0], 1)
        dst.write(rgb_uint8[:, :, 1], 2)
        dst.write(rgb_uint8[:, :, 2], 3)


def write_confidence_geotiff(path: Path, confidence: np.ndarray, meta: Dict) -> None:
    out_meta = meta.copy()
    out_meta.update(
        count=1,
        dtype="uint8",
        compress="lzw",
    )

    confidence_uint8 = (np.clip(confidence, 0.0, 1.0) * 255).astype(np.uint8)

    with rasterio.open(path, "w", **out_meta) as dst:
        dst.write(confidence_uint8, 1)


def write_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    rgb_uint8 = rgb_to_uint8(rgb)
    img = Image.fromarray(rgb_uint8)
    img.save(path)
