"""3D reprojection, point-cloud construction, and PLY export."""
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def reproject_to_3d(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Reproject a disparity map to 3D coordinates.

    Inputs: disparity (H, W) in pixels, 4x4 Q reprojection matrix from
        stereoRectify (depth in mm given T in mm).
    Outputs: (H, W, 3) float32 array of (X, Y, Z) coordinates in mm.
    """
    points = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q)
    logger.info("Reprojected to 3D: %s", points.shape)
    return points.astype(np.float32)


def build_point_cloud(points_3d: np.ndarray, color_image: np.ndarray,
                      disparity: np.ndarray, min_disparity: float = 1.0):
    """Build a filtered colored point cloud.

    Inputs:
        points_3d: (H, W, 3) reprojected points (mm).
        color_image: (H, W, 3) BGR uint8 image for point colors.
        disparity: (H, W) float32 disparity in pixels.
        min_disparity: keep points with disparity strictly above this (px).
    Outputs: (points (N, 3) float32 in mm, colors (N, 3) uint8 RGB in 0-255).
    Points with disparity <= min_disparity or non-finite coordinates are removed.
    """
    if (disparity.ndim != 2 or points_3d.shape != disparity.shape + (3,)
            or color_image.shape != points_3d.shape):
        raise ValueError("Point, color, and disparity image shapes must agree")
    colors_bgr = color_image
    mask = ((disparity > min_disparity) & np.isfinite(disparity)
            & np.isfinite(points_3d).all(axis=2))
    points = points_3d[mask].astype(np.float32)
    colors = cv2.cvtColor(colors_bgr, cv2.COLOR_BGR2RGB)[mask].astype(np.uint8)
    logger.info("Point cloud: %d points kept of %d", points.shape[0], mask.size)
    return points, colors


def cap_depth_outliers(depth: np.ndarray, percentile: float = 99.0):
    """Cap far-depth outliers using a percentile (Module 5 smoothness prior).

    Inputs: depth (H, W) float32 in mm (NaN/inf = invalid), percentile in (0, 100).
    Outputs: (capped depth with far tails set to NaN, cap value or None).
    Syllabus: percentile clipping is histogram processing (Module 1).
    """
    if not 0.0 < percentile < 100.0:
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    capped = np.asarray(depth, dtype=np.float32).copy()
    finite = capped[np.isfinite(capped)]
    if finite.size == 0:
        return capped, None
    cap = float(np.percentile(finite, percentile))
    capped[np.isfinite(capped) & (capped > cap)] = np.nan
    logger.info("Depth capped at %.1f mm (%.1fth percentile, %d finite)",
                cap, percentile, finite.size)
    return capped, cap


def clean_point_cloud(points: np.ndarray, colors: np.ndarray,
                      z_percentile: float = 99.0, std_ratio: float = 2.5):
    """Remove far-tail and statistical outliers (Modules 4-5, classical only).

    Inputs: points (N, 3) float32 mm, colors (N, 3) uint8, Z percentile cap,
        per-axis median +/- std_ratio * std gate.
    Outputs: filtered (points, colors); empty input returns empty output.
    No learning / NeRF / deep models: percentile + mean/std gating only.
    """
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors)
    if points.shape[0] == 0:
        return points.reshape(0, 3).astype(np.float32), colors.reshape(0, 3)
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("points and RGB colors must have matching (N, 3) shapes")
    finite = np.isfinite(points).all(axis=1)
    points, colors = points[finite], colors[finite]
    if points.shape[0] == 0:
        return points.astype(np.float32), colors
    if not 0.0 < z_percentile < 100.0:
        raise ValueError(f"z_percentile must be in (0, 100), got {z_percentile}")
    if not np.isfinite(std_ratio) or std_ratio <= 0:
        raise ValueError(f"std_ratio must be positive, got {std_ratio}")
    z_cap = float(np.percentile(points[:, 2], z_percentile))
    keep = points[:, 2] <= z_cap
    points, colors = points[keep], colors[keep]
    if points.shape[0] == 0:
        return points.astype(np.float32), colors
    median = np.median(points.astype(np.float64), axis=0)
    std = np.std(points.astype(np.float64), axis=0) + 1e-6
    keep = (np.abs(points.astype(np.float64) - median) <= std_ratio * std).all(axis=1)
    points, colors = points[keep], colors[keep]
    logger.info("Cleaned cloud: %d points kept (Z cap %.1f mm, std_ratio %.2f)",
                points.shape[0], z_cap, std_ratio)
    return points.astype(np.float32), colors


def save_ply(path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """Write a binary-little-endian PLY point cloud file.

    Inputs: output path, points (N, 3) float32 (mm), colors (N, 3) uint8 (0-255).
    Outputs: None; raises IOError on write failure.
    """
    if (points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape
            or not np.isfinite(points).all() or not np.isfinite(colors).all()
            or np.any(colors < 0) or np.any(colors > 255)):
        raise ValueError("PLY requires finite (N, 3) points and matching RGB colors in 0..255")
    if points.shape[0] != colors.shape[0]:
        raise ValueError(
            f"points/colors count mismatch: {points.shape[0]} vs {colors.shape[0]}"
        )
    n = points.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    vertex_data = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vertex_data["x"], vertex_data["y"], vertex_data["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex_data["r"], vertex_data["g"], vertex_data["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vertex_data.tobytes())
    logger.info("Saved PLY with %d vertices: %s", n, path)
