"""Single-camera chessboard calibration."""
import logging
import os

import cv2
import numpy as np

from src.errors import CalibrationError

logger = logging.getLogger(__name__)


def _find_corners(img_gray, pattern_size):
    """Detect and refine chessboard corners in one image.

    Inputs: grayscale image, (cols, rows) internal corner count.
    Outputs: (Nx1x2 float32 corners, (w, h)) or (None, (w, h)) if not found.
    """
    h, w = img_gray.shape
    found, corners = cv2.findChessboardCorners(
        img_gray, pattern_size,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None, (w, h)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    corners = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)
    return corners, (w, h)


def calibrate_camera(image_paths: list, pattern_size: tuple, square_size_mm: float) -> dict:
    """Calibrate one camera from chessboard images.

    Inputs:
        image_paths: list of image file paths (>= 12 with detectable corners).
        pattern_size: (cols, rows) internal corners of the chessboard.
        square_size_mm: physical square size in millimeters.
    Outputs: dict with keys:
        "K" (3x3 ndarray, px), "dist" (distortion coefficients),
        "rms_error" (px), "image_size" ((w, h) px).
    Raises CalibrationError if fewer than 12 images yield corners.
    """
    if pattern_size[0] <= 0 or pattern_size[1] <= 0:
        raise CalibrationError(f"Invalid chessboard pattern_size: {pattern_size}")
    if square_size_mm <= 0:
        raise CalibrationError(f"Invalid square_size_mm: {square_size_mm}")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints, imgpoints = [], []
    image_size = None
    used = 0

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning("Could not read image, skipping: %s", path)
            continue
        corners, size = _find_corners(img, tuple(pattern_size))
        if corners is None:
            logger.warning("Chessboard corners not found, skipping: %s", path)
            continue
        if image_size is None:
            image_size = size
        elif size != image_size:
            logger.warning("Image size mismatch (%s vs %s), skipping: %s",
                           size, image_size, path)
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        used += 1

    min_images = 12
    if used < min_images:
        raise CalibrationError(
            f"Only {used} of {len(image_paths)} images yielded usable corners; "
            f"need at least {min_images}."
        )
    logger.info("Calibrating with %d/%d images", used, len(image_paths))

    rms, K, dist, _, _ = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    logger.info("Calibration RMS reprojection error: %.4f px", rms)
    return {"K": K, "dist": dist, "rms_error": rms, "image_size": image_size}


def list_images(directory: str) -> list:
    """List supported image files in a directory (sorted).

    Inputs: directory path.
    Outputs: list of file paths.
    """
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Calibration image directory not found: {directory}")
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(exts)
    )
