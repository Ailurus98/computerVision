"""Static (matplotlib) and optional interactive (Open3D) visualizations."""
import logging

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

logger = logging.getLogger(__name__)


def show_disparity_heatmap(disparity: np.ndarray, save_path: str) -> None:
    """Render a disparity map as a 'jet' heatmap and save it to PNG.

    Inputs: disparity (H, W) float32 in px, output PNG path.
    Outputs: None; writes save_path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(disparity, cmap="jet")
    fig.colorbar(im, ax=ax, label="disparity (px)")
    ax.set_title("Disparity Map (SGBM)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved disparity heatmap: %s", save_path)


def show_point_cloud_preview(points: np.ndarray, colors: np.ndarray, save_path: str) -> None:
    """Render a static 3D scatter preview of a point cloud and save it to PNG.

    Inputs: points (N, 3) in mm, colors (N, 3) uint8 RGB 0-255, output PNG path.
    Outputs: None; writes save_path.
    """
    step = max(1, points.shape[0] // 50000)
    pts, cols = points[::step], colors[::step]
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=cols / 255.0, s=0.5, marker=".")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_zlabel("Y (mm)")
    ax.set_title("3D Point Cloud Preview")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info("Saved point cloud preview: %s", save_path)


def show_point_cloud_interactive(points: np.ndarray, colors: np.ndarray) -> None:
    """Optionally display an interactive Open3D point cloud (no-op without open3d).

    Inputs: points (N, 3) in mm, colors (N, 3) uint8 RGB 0-255.
    Outputs: None; raises ImportError if open3d is not installed.
    """
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is not installed; install it for the interactive viewer"
        ) from exc
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.visualization.draw_geometries([pcd])
