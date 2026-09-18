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
        "mode": cv2.StereoSGBM_MODE_SGBM_3WAY,
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
               "disp12MaxDiff", "uniquenessRatio", "speckleWindowSize", "speckleRange",
               "mode"}
    if merged.keys() - allowed:
        raise ValueError(f"Unknown SGBM parameters: {sorted(merged.keys() - allowed)}")
    matcher = cv2.StereoSGBM_create(**merged)
    return matcher.compute(rect_left_gray, rect_right_gray).astype(np.float32) / 16.0


def compute_disparity_filtered(left, right, params: dict) -> tuple:
    """Return (raw, filtered, validmask) on the left image's pixel grid.

    Disparities and thresholds are in pixels; arrays are float32 and mask is bool.
    lr_max_diff defaults to 1 px. median_max_diff defaults to 2 px; None disables
    the 3x3 median residual test. Invalid filtered pixels are NaN, never filled.
    Base SGBM speckle settings apply independently in both matching directions.
    """
    base = dict(params)
    lr_max_diff = base.pop("lr_max_diff", 1.0)
    median_max_diff = base.pop("median_max_diff", 2.0)
    for name, value in (("lr_max_diff", lr_max_diff),
                        ("median_max_diff", median_max_diff)):
        if name == "median_max_diff" and value is None:
            continue
        if not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    raw = compute_disparity(left, right, base)
    min_left = base.get("minDisparity", 0)
    dimensions = {"numDisparities": value for key, value in base.items()
                  if key in ("numDisparities", "num_disparities")}
    if not dimensions:
        defaults = load_config(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
        dimensions["numDisparities"] = defaults["stereo"]["num_disparities"]
    min_right = -min_left - dimensions["numDisparities"] + 1
    reverse = compute_disparity(right, left, {**base, "minDisparity": min_right})
    valid_left = np.isfinite(raw) & (raw > min_left - 1)
    y, x = np.indices(raw.shape)
    sample_x = np.rint(x - np.where(valid_left, raw, 0))
    validmask = valid_left & (sample_x >= 0) & (sample_x < raw.shape[1])
    sample_x = np.clip(sample_x, 0, raw.shape[1] - 1).astype(np.intp)
    sampled = reverse[y, sample_x]
    validmask &= np.isfinite(sampled) & (sampled > min_right - 1)
    residual = np.where(validmask, raw, 0) + np.where(validmask, sampled, 0)
    validmask &= np.abs(residual) <= lr_max_diff
    if median_max_diff is not None and validmask.any():
        padded = np.pad(np.where(valid_left, raw, np.nan), 1, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        median = np.nanmedian(windows[validmask], axis=(-2, -1))
        validmask[validmask] = np.abs(raw[validmask] - median) <= median_max_diff
    filtered = np.where(validmask, raw, np.nan).astype(np.float32)
    return raw, filtered, validmask
