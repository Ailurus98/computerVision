"""Disparity estimation via Semi-Global Block Matching."""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "minDisparity": 0,
    "numDisparities": 128,
    "blockSize": 5,
    "disp12MaxDiff": 1,
    "uniquenessRatio": 10,
    "speckleWindowSize": 100,
    "speckleRange": 2,
}


def compute_disparity(rect_left_gray, rect_right_gray, params: dict) -> np.ndarray:
    """Compute a dense disparity map from a rectified stereo pair.

    Inputs: left/right grayscale rectified images (uint8), params dict
        (defaults in DEFAULT_PARAMS; missing keys fall back to defaults).
    Outputs: float32 disparity map (H, W), values in pixels (fixed-point
        scale of 16 removed).
    Raises ValueError if numDisparities is not a positive multiple of 16.
    """
    merged = dict(DEFAULT_PARAMS)
    merged.update(params)
    nd = merged["numDisparities"]
    bs = merged["blockSize"]
    if not isinstance(nd, int) or nd <= 0 or nd % 16 != 0:
        raise ValueError(
            f"numDisparities must be a positive multiple of 16, got {nd!r}"
        )
    if not isinstance(bs, int) or bs < 3 or bs % 2 == 0:
        raise ValueError(f"blockSize must be an odd integer >= 3, got {bs!r}")

    merged.setdefault("P1", 8 * 3 * bs ** 2)
    merged.setdefault("P2", 32 * 3 * bs ** 2)

    matcher = cv2.StereoSGBM_create(
        minDisparity=merged["minDisparity"],
        numDisparities=merged["numDisparities"],
        blockSize=merged["blockSize"],
        P1=merged["P1"],
        P2=merged["P2"],
        disp12MaxDiff=merged["disp12MaxDiff"],
        uniquenessRatio=merged["uniquenessRatio"],
        speckleWindowSize=merged["speckleWindowSize"],
        speckleRange=merged["speckleRange"],
    )
    logger.info("Computing disparity (SGBM) on shape %s", rect_left_gray.shape)
    raw = matcher.compute(rect_left_gray, rect_right_gray)
    disparity = raw.astype(np.float32) / 16.0
    logger.info(
        "Disparity computed: shape=%s range=[%.2f, %.2f]",
        disparity.shape, float(np.min(disparity)), float(np.max(disparity)),
    )
    return disparity


def refine_disparity_wls(disparity, rect_left_gray, rect_right_gray, lam=8000.0, sigma=1.5) -> np.ndarray:
    """Optional WLS-filter refinement of the disparity map (not used by default).

    Inputs: SGBM disparity (float32), left/right grayscale rectified images.
    Outputs: refined float32 disparity map. Requires opencv-contrib-python.
    """
    right_matcher = cv2.ximgproc.createRightMatcher(
        cv2.StereoSGBM_create(
            minDisparity=DEFAULT_PARAMS["minDisparity"],
            numDisparities=DEFAULT_PARAMS["numDisparities"],
            blockSize=DEFAULT_PARAMS["blockSize"],
        )
    )
    disp_r = right_matcher.compute(rect_right_gray, rect_left_gray).astype(np.float32) / 16.0
    wls = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
    wls.setLambda(lam)
    wls.setSigmaColor(sigma)
    return wls.filter(disparity, rect_left_gray, disparity_map_right=disp_r)
