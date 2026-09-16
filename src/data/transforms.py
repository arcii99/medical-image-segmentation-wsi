"""Augmentation (ADR-008).

Split into two pipelines on purpose.  Geometric transforms apply to image and
mask; photometric transforms apply to the image ONLY.  Routing a colour
transform to the mask silently destroys the labels (BUG-011), so the split is
structural rather than a convention.
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["geometric", "photometric", "normalize", "TrainTransform", "EvalTransform"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# H&E stain vectors (Ruifrok & Johnston), rows = H, E, DAB
_HED_FROM_RGB = np.linalg.inv(np.array([
    [0.65, 0.70, 0.29],
    [0.07, 0.99, 0.11],
    [0.27, 0.57, 0.78],
], np.float32))
_RGB_FROM_HED = np.linalg.inv(_HED_FROM_RGB)


def hed_jitter(img: np.ndarray, rng: np.random.Generator,
               alpha: float = 0.05, beta: float = 0.05) -> np.ndarray:
    """Perturb haematoxylin and eosin channels in optical-density space.

    Preferred over test-time stain normalisation: no reference image to
    version, no per-patch SVD, and better cross-scanner generalisation.
    """
    x = img.astype(np.float32) / 255.0
    od = -np.log(np.clip(x, 1e-6, 1.0))
    hed = od.reshape(-1, 3) @ _HED_FROM_RGB
    a = rng.uniform(1 - alpha, 1 + alpha, 3).astype(np.float32)
    b = rng.uniform(-beta, beta, 3).astype(np.float32)
    hed = hed * a + b
    od = (hed @ _RGB_FROM_HED).reshape(img.shape)
    return np.clip(np.exp(-od) * 255.0, 0, 255).astype(np.uint8)


def geometric(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    """D4 dihedral group. Applied to image AND mask."""
    k = int(rng.integers(0, 4))
    if k:
        img, mask = np.rot90(img, k, (0, 1)), np.rot90(mask, k, (0, 1))
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[::-1], mask[::-1]
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Image ONLY. Never receives the mask (BUG-011)."""
    if rng.random() < 0.8:
        img = hed_jitter(img, rng)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.3:
        q = int(rng.integers(55, 95))
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return img


def normalize(img: np.ndarray) -> np.ndarray:
    """uint8 HWC -> float32 CHW, ImageNet-normalised."""
    x = img.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


class TrainTransform:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        img, mask = geometric(img, mask, self.rng)
        img = photometric(img, self.rng)
        mask = (mask > 0).astype(np.float32)      # post-condition, BUG-011/013
        return normalize(img), mask[None]


class EvalTransform:
    def __call__(self, img: np.ndarray, mask: np.ndarray):
        return normalize(img), (mask > 0).astype(np.float32)[None]
