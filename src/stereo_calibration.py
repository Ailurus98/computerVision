"""Stereo calibration, rectification, and rectified-image application."""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def stereo_calibrate(K1, dist1, K2, dist2, objpoints, imgpoints_l, imgpoints_r, image_size) -> dict:
    """Jointly calibrate a stereo camera pair.

    Inputs: per-camera intrinsics (K in px, distortion coeffs), matched
        object/image points from calibration chessboards, image size (w, h).
    Outputs: dict with "R" (3x3), "T" (3x1, mm), "E", "F".
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv2.CALIB_FIX_INTRINSIC
    rms, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        K1, dist1, K2, dist2, image_size,
        criteria=criteria, flags=flags,
    )
    logger.info("Stereo calibration RMS reprojection error: %.4f px", rms)
    return {"R": R, "T": T, "E": E, "F": F}


def rectify(K1, dist1, K2, dist2, R, T, image_size) -> dict:
    """Compute rectification maps and the Q reprojection matrix.

    Inputs: intrinsics/distortions, stereo rotation R and translation T (mm),
        image size (w, h).
    Outputs: dict with "map_l1", "map_l2", "map_r1", "map_r2" (remap inputs)
        and "Q" (4x4 reprojection matrix, depth in mm).
    """
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1, dist1, K2, dist2, image_size, R, T,
        alpha=0,
    )
    map_l1, map_l2 = cv2.initUndistortRectifyMap(K1, dist1, R1, P1, image_size, cv2.CV_32FC1)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(K2, dist2, R2, P2, image_size, cv2.CV_32FC1)
    logger.info("Rectification maps and Q matrix computed for size %s", image_size)
    return {
        "map_l1": map_l1, "map_l2": map_l2,
        "map_r1": map_r1, "map_r2": map_r2,
        "Q": Q,
    }


def apply_rectification(img_left, img_right, maps):
    """Apply rectification maps to a stereo pair.

    Inputs: left/right images (same size), maps dict from rectify().
    Outputs: (rect_left, rect_right) rectified images.
    """
    rect_left = cv2.remap(img_left, maps["map_l1"], maps["map_l2"], cv2.INTER_LINEAR)
    rect_right = cv2.remap(img_right, maps["map_r1"], maps["map_r2"], cv2.INTER_LINEAR)
    return rect_left, rect_right
