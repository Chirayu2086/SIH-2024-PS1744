from typing import Dict, Tuple

import cv2
import numpy as np
import rasterio


FEATURE_NAMES = [
    "band_1",
    "band_2",
    "ratio_or_copy",
    "span_or_local_mean",
    "local_std",
    "edge_x",
    "edge_y",
    "texture_contrast",
    "texture_energy",
    "terrain_or_blank",
]


def read_geotiff(path: str) -> Tuple[np.ndarray, Dict]:
    with rasterio.open(path) as src:
        img = src.read().astype(np.float32)
        meta = src.meta.copy()

    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return img, meta


def percentile_normalize(arr: np.ndarray, low: float = 2.0, high: float = 98.0) -> np.ndarray:
    if arr.size == 0:
        return arr

    p_low = np.percentile(arr, low)
    p_high = np.percentile(arr, high)

    if p_high - p_low < 1e-6:
        return np.zeros_like(arr)

    normalized = (arr - p_low) / (p_high - p_low)
    return np.clip(normalized, 0.0, 1.0)


def local_mean(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.blur(img, (ksize, ksize))


def local_std(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    mean = local_mean(img, ksize)
    sq_mean = local_mean(img * img, ksize)
    variance = np.maximum(sq_mean - mean * mean, 0.0)
    return np.sqrt(variance)


def sobel_edges(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img_uint8 = (img * 255).astype(np.uint8)
    gx = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)

    gx = percentile_normalize(np.abs(gx))
    gy = percentile_normalize(np.abs(gy))

    return gx, gy


def texture_features(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img_uint8 = (img * 255).astype(np.uint8)

    blur = cv2.GaussianBlur(img_uint8, (3, 3), 0)
    contrast = np.abs(img_uint8.astype(np.float32) - blur.astype(np.float32))
    contrast = percentile_normalize(contrast)

    energy = cv2.blur(img_uint8.astype(np.float32), (5, 5))
    energy = percentile_normalize(energy)

    return contrast, energy


def build_feature_stack(img: np.ndarray) -> np.ndarray:
    """
    Build a fixed-size feature stack for the model.

    Input:
        img: shape (bands, H, W)

    Output:
        features: shape (10, H, W)
    """
    bands, h, w = img.shape
    features = []

    band1 = percentile_normalize(img[0])
    features.append(band1)

    if bands >= 2:
        band2 = percentile_normalize(img[1])
        eps = 1e-6
        ratio = band1 / (band2 + eps)
        ratio = percentile_normalize(ratio)
        span = percentile_normalize(band1 + band2)
    else:
        band2 = band1.copy()
        ratio = local_mean(band1)
        span = local_mean(band1)

    features.append(band2)
    features.append(ratio)
    features.append(span)

    lstd = local_std(band1)
    lstd = percentile_normalize(lstd)
    features.append(lstd)

    edge_x, edge_y = sobel_edges(band1)
    features.append(edge_x)
    features.append(edge_y)

    texture_contrast, texture_energy = texture_features(band1)
    features.append(texture_contrast)
    features.append(texture_energy)

    # Placeholder terrain channel.
    # In production, replace with DEM slope/hillshade/local incidence.
    terrain = np.zeros((h, w), dtype=np.float32)
    features.append(terrain)

    while len(features) < len(FEATURE_NAMES):
        features.append(np.zeros((h, w), dtype=np.float32))

    features = features[: len(FEATURE_NAMES)]
    return np.stack(features, axis=0).astype(np.float32)
