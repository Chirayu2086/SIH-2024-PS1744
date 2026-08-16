import cv2
import numpy as np


def apply_gamma(rgb: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    return np.clip(rgb ** (1.0 / max(gamma, 1e-6)), 0.0, 1.0)


def natural_mode(rgb: np.ndarray) -> np.ndarray:
    return apply_gamma(rgb, gamma=1.0)


def landcover_mode(rgb: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    """
    Simple heuristic palette.
    In production, replace with trained landcover conditioning.
    """
    h, w, _ = rgb.shape
    output = rgb.copy()

    brightness = rgb.mean(axis=2)

    vegetation_mask = brightness > 0.55
    water_mask = brightness < 0.25
    built_mask = (brightness >= 0.25) & (brightness <= 0.55)

    output[vegetation_mask] = [0.20, 0.65, 0.25]
    output[water_mask] = [0.10, 0.35, 0.80]
    output[built_mask] = [0.75, 0.55, 0.35]

    output *= confidence[:, :, None]
    return np.clip(output, 0.0, 1.0)


def geology_mode(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """
    Edge-enhanced geology interpretation mode.
    This is not true color. It is a structural enhancement.
    """
    gray_uint8 = (gray * 255).astype(np.uint8)

    blurred = cv2.GaussianBlur(gray_uint8, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 140)

    gx = cv2.Sobel(gray_uint8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_uint8, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = np.clip(magnitude / max(magnitude.max(), 1e-6), 0.0, 1.0)

    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    base = cv2.cvtColor(gray_uint8, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0

    edge_overlay = base.copy()
    edge_overlay[edges > 0] = [1.0, 0.75, 0.10]

    slope_overlay = edge_overlay.copy()
    slope_overlay[:, :, 0] += magnitude * 0.25
    slope_overlay[:, :, 1] += magnitude * 0.10

    return np.clip(slope_overlay, 0.0, 1.0)


def water_mode(rgb: np.ndarray, gray: np.ndarray) -> np.ndarray:
    output = rgb.copy()

    dark_mask = gray < 0.25
    texture_mask = gray > 0.70

    output[dark_mask] = [0.05, 0.35, 0.90]
    output[texture_mask] = [0.90, 0.25, 0.20]

    return np.clip(output, 0.0, 1.0)
