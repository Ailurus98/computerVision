"""Disparity-to-depth conversion."""
import logging

import numpy as np

logger = logging.getLogger(__name__)


def disparity_to_depth(disparity: np.ndarray, focal_length_px: float, baseline_mm: float) -> np.ndarray:
    """Convert a disparity map to metric depth via Z = f * B / d.

    Inputs:
        disparity: float32 array (H, W), values in pixels.
        focal_length_px: focal length in pixels (from calibration K, e.g. K1[0, 0]).
        baseline_mm: camera baseline in millimeters (from stereo translation T).
    Outputs: float32 depth map (H, W) in millimeters; d <= 0 maps to np.inf.
    """
    if baseline_mm <= 0:
        raise ValueError(f"baseline_mm must be positive, got {baseline_mm}")
    if focal_length_px <= 0:
        raise ValueError(f"focal_length_px must be positive, got {focal_length_px}")
    disparity = np.asarray(disparity, dtype=np.float32)
    depth = np.full(disparity.shape, np.inf, dtype=np.float32)
    valid = disparity > 0
    depth[valid] = (focal_length_px * baseline_mm) / disparity[valid]
    logger.info(
        "Depth computed: shape=%s valid=%.1f%%",
        depth.shape, 100.0 * float(np.count_nonzero(valid)) / disparity.size,
    )
    return depth
