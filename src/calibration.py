"""Single-camera chessboard calibration with millimeter object coordinates."""
import logging
import os
from pathlib import Path

import cv2
import numpy as np

from src.errors import CalibrationError
from src.utils import load_config, load_image_gray

logger = logging.getLogger(__name__)


def _find_corners(img_gray, pattern_size):
    found, corners = cv2.findChessboardCorners(
        img_gray, pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    size = (img_gray.shape[1], img_gray.shape[0])
    if not found:
        return None, size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
    return corners, size


def _object_grid(pattern_size, square_size_mm):
    if (len(pattern_size) != 2 or
            any(type(n) is not int or n < 2 for n in pattern_size)):
        raise CalibrationError(f"Invalid pattern_size: {pattern_size}")
    if not np.isfinite(square_size_mm) or square_size_mm <= 0:
        raise CalibrationError(f"Invalid square_size_mm: {square_size_mm}")
    grid = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    grid[:, :2] = np.mgrid[:pattern_size[0], :pattern_size[1]].T.reshape(-1, 2)
    return grid * square_size_mm


def _minimum_images(min_images=None):
    path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    minimum = load_config(path)["calibration"]["min_images"]
    if min_images is None:
        return minimum
    if type(min_images) is not int or min_images < minimum:
        raise ValueError(f"min_images must be an integer >= {minimum}")
    return min_images


def _collect_corners(paths, pattern_size):
    detections, image_size = {}, None
    for path in paths:
        gray = load_image_gray(path)
        corners, size = _find_corners(gray, tuple(pattern_size))
        if corners is None:
            logger.warning("Chessboard corners not found, skipping: %s", path)
            continue
        if image_size is not None and size != image_size:
            raise ValueError(f"Calibration image size mismatch: {path}: {size} != {image_size}")
        image_size = size
        detections[Path(path).name] = corners
    return detections, image_size


def calibrate_camera(image_paths: list[str], pattern_size: tuple[int, int],
                     square_size_mm: float, *, min_images=None) -> dict:
    """Input image paths, corner counts, square size (mm), optional minimum count.

    Return K (3x3, px), dist (unitless), rms_error (px), image_size (w, h px).
    Insufficient successful detections raise CalibrationError; minimum comes from YAML.
    """
    grid = _object_grid(pattern_size, square_size_mm)
    minimum = _minimum_images(min_images)
    detections, image_size = _collect_corners(image_paths, pattern_size)
    if len(detections) < minimum:
        raise CalibrationError(
            f"Only {len(detections)} images yielded corners; need at least {minimum}"
        )
    logger.info("Camera calibration start: corners shape=%s", np.shape(list(detections.values())))
    rms, K, dist, _, _ = cv2.calibrateCamera(
        [grid] * len(detections), list(detections.values()), image_size, None, None
    )
    logger.info("Camera calibration end: K shape=%s RMS=%.4f px", K.shape, rms)
    return {"K": K, "dist": dist, "rms_error": float(rms), "image_size": image_size}


def list_images(directory: str) -> list[str]:
    """Input directory path; return sorted static image paths (no physical units)."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Calibration image directory not found: {directory}")
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted(str(p) for p in Path(directory).iterdir()
                  if p.is_file() and p.suffix.lower() in extensions)


def collect_stereo_points(left_paths, right_paths, pattern_size, square_size_mm,
                          min_images=None):
    """Input same-filename synchronized views, corner counts, square size (mm).

    Return object points (mm), left/right corners (px), and image size (w, h px).
    A pair is discarded together if either image lacks detected corners.
    """
    grid = _object_grid(pattern_size, square_size_mm)
    minimum = _minimum_images(min_images)
    left = {Path(p).name: p for p in left_paths}
    right = {Path(p).name: p for p in right_paths}
    if left.keys() != right.keys():
        raise CalibrationError("Stereo calibration requires matching filenames for synchronized pairs")
    dl, size_l = _collect_corners(left_paths, pattern_size)
    dr, size_r = _collect_corners(right_paths, pattern_size)
    if size_l != size_r:
        raise ValueError(f"Calibration camera sizes differ: {size_l} vs {size_r}")
    names = sorted(dl.keys() & dr.keys())
    if len(names) < minimum:
        raise CalibrationError(f"Only {len(names)} usable stereo pairs; need at least {minimum}")
    return [grid] * len(names), [dl[n] for n in names], [dr[n] for n in names], size_l
