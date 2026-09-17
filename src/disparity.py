"""CPU-only Semi-Global Block Matching with disparities in pixels."""
from pathlib import Path

import cv2
import numpy as np

from src.utils import load_config, validate_image_pair_shapes


def compute_disparity(rect_left_gray, rect_right_gray, params: dict) -> np.ndarray:
    """Input matching uint8 grayscale images and SGBM settings; return float32 px.

    YAML snake_case num_disparities/block_size and OpenCV names are accepted.
    Missing dimensions come from config/config.yaml, other defaults match the spec.
    """
    merged = {
        "minDisparity": 0, "disp12MaxDiff": 1, "uniquenessRatio": 10,
        "speckleWindowSize": 100, "speckleRange": 2,
    }
    aliases = {"num_disparities": "numDisparities", "block_size": "blockSize"}
    for key, value in params.items():
        merged[aliases.get(key, key)] = value
    if "numDisparities" not in merged or "blockSize" not in merged:
        defaults = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
        for key, alias in aliases.items():
            merged.setdefault(alias, defaults["stereo"][key])
    nd, bs = merged["numDisparities"], merged["blockSize"]
    if type(nd) is not int or nd <= 0 or nd % 16:
        raise ValueError(f"numDisparities must be a positive multiple of 16, got {nd!r}")
    if type(bs) is not int or bs <= 0 or bs % 2 == 0:
        raise ValueError(f"blockSize must be a positive odd integer, got {bs!r}")
    validate_image_pair_shapes(rect_left_gray, rect_right_gray)
    if (rect_left_gray.ndim != 2 or rect_left_gray.dtype != np.uint8
            or rect_right_gray.dtype != np.uint8 or rect_left_gray.size == 0):
        raise ValueError("SGBM requires nonempty uint8 grayscale images")
    if type(merged["minDisparity"]) is not int:
        raise ValueError("minDisparity must be an integer")
    if rect_left_gray.shape[1] <= nd + max(merged["minDisparity"], 0) + bs:
        raise ValueError("Image width is too small for the configured disparity search")
    merged.setdefault("P1", 8 * 3 * bs ** 2)
    merged.setdefault("P2", 32 * 3 * bs ** 2)
    if merged["P1"] < 0 or merged["P2"] <= merged["P1"]:
        raise ValueError("SGBM requires 0 <= P1 < P2")
    allowed = {"minDisparity", "numDisparities", "blockSize", "P1", "P2",
               "disp12MaxDiff", "uniquenessRatio", "speckleWindowSize", "speckleRange"}
    if merged.keys() - allowed:
        raise ValueError(f"Unknown SGBM parameters: {sorted(merged.keys() - allowed)}")
    matcher = cv2.StereoSGBM_create(**merged)
    return matcher.compute(rect_left_gray, rect_right_gray).astype(np.float32) / 16.0
