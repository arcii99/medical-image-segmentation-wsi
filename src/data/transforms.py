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


# Optical density of every possible uint8 value, precomputed.
#
# The forward transform starts with -log(x/255) over 786k pixels. The input is
# uint8, so there are only 256 distinct results: a lookup replaces the log
# entirely. Measured 12.7 ms -> 4.6 ms per patch, which matters because stain
# jitter was 62% of the data-loading cost and data loading was the bottleneck
# on a 2-core VM.
_OD_LUT = -np.log(np.clip(np.arange(256, dtype=np.float32) / 255.0, 1e-6, 1.0))


def hed_jitter(img: np.ndarray, rng: np.random.Generator,
               alpha: float = 0.05, beta: float = 0.05) -> np.ndarray:
    """Perturb haematoxylin and eosin channels in optical-density space.

    Preferred over test-time stain normalisation: no reference image to
    version, no per-patch SVD, and better cross-scanner generalisation.

    Algebraically this is a single affine map in OD space. Writing it that way
    collapses two 3x3 matmuls over 786k pixels into one:

        hed      = od @ H
        hed'     = hed * a + b
        od'      = hed' @ H^-1
                 = od @ (H diag(a) H^-1) + (b @ H^-1)
                 = od @ M + c
    """
    a = rng.uniform(1 - alpha, 1 + alpha, 3).astype(np.float32)
    b = rng.uniform(-beta, beta, 3).astype(np.float32)
    M = (_HED_FROM_RGB * a) @ _RGB_FROM_HED          # 3x3
    c = b @ _RGB_FROM_HED                            # 3

    od = _OD_LUT[img]                                # LUT instead of log
    od = od.reshape(-1, 3) @ M + c
    np.exp(-od, out=od)
    od *= 255.0
    return np.clip(od, 0, 255).astype(np.uint8).reshape(img.shape)


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


def photometric(img: np.ndarray, rng: np.random.Generator,
                p_stain: float = 0.8, p_blur: float = 0.3,
                p_jpeg: float = 0.0) -> np.ndarray:
    """Image ONLY. Never receives the mask (BUG-011).

    ``p_jpeg`` defaults to 0. Simulating compression artifacts is worthwhile
    on lossless source data, but an exported patch set is already JPEG q90 --
    re-encoding it a second time models nothing the model has not already
    seen, and costs 3.3 ms per patch on the critical path.
    """
    if rng.random() < p_stain:
        img = hed_jitter(img, rng)
    if rng.random() < p_blur:
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.3, 1.2)))
    if p_jpeg and rng.random() < p_jpeg:
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
    """Augmentation probabilities are configurable because their cost is not
    negligible: stain jitter is ~8.6 ms per patch, and on a 2-core machine
    the data pipeline can become the bottleneck rather than the GPU. Lowering
    p_stain is the cheapest way to buy throughput back, at a real cost in
    stain-invariance (ADR-008)."""

    def __init__(self, seed: int = 0, p_stain: float = 0.8,
                 p_blur: float = 0.3, p_jpeg: float = 0.0):
        self.rng = np.random.default_rng(seed)
        self.p_stain, self.p_blur, self.p_jpeg = p_stain, p_blur, p_jpeg

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        img, mask = geometric(img, mask, self.rng)
        img = photometric(img, self.rng, self.p_stain, self.p_blur,
                          self.p_jpeg)
        mask = (mask > 0).astype(np.float32)      # post-condition, BUG-011/013
        return normalize(img), mask[None]


class EvalTransform:
    def __call__(self, img: np.ndarray, mask: np.ndarray):
        return normalize(img), (mask > 0).astype(np.float32)[None]
