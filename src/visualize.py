"""Headless PNG exports and optional interactive matplotlib visualizations."""
import logging
import os
import sys

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


def enable_interactive() -> bool:
    """Probe for a working GUI backend; return True or log a warning and False."""
    import matplotlib

    errors = (ImportError, RuntimeError, ValueError, OSError)
    try:
        from tkinter import TclError
    except ImportError:
        tk_available = False
    else:
        errors += (TclError,)
        tk_available = True
    original = None
    try:
        import matplotlib.pyplot as plt
        original = matplotlib.get_backend()
        try:
            from matplotlib.backends import backend_registry
        except ImportError:
            from matplotlib.rcsetup import interactive_bk

            interactive = interactive_bk
        else:
            from matplotlib.backends import BackendFilter

            interactive = backend_registry.list_builtin(BackendFilter.INTERACTIVE)
        candidates = [original] if original.lower() in {b.lower() for b in interactive} else []
        display = sys.platform in ("win32", "darwin") or bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        if tk_available and display and original.lower() != "tkagg":
            candidates.append("TkAgg")
        for backend in candidates:
            fig = None
            try:
                if matplotlib.get_backend().lower() != backend.lower():
                    plt.switch_backend(backend)
                fig = plt.figure()
                fig.canvas.draw()
                logger.info("Using interactive backend: %s", backend)
                return True
            except errors as exc:
                logger.warning("Interactive backend %s failed: %s", backend, exc)
            finally:
                if fig is not None:
                    plt.close(fig)
    except errors as exc:
        logger.warning("Cannot initialize interactive matplotlib: %s", exc)
    if original is not None:
        try:
            plt.switch_backend(original)
        except errors:
            pass
    logger.warning("Interactive display unavailable; check the display and GUI backend (TkAgg needs tkinter).")
    return False


def _draw_cloud(ax, points, colors, title, max_points):
    pts = np.asarray(points, dtype=np.float64)
    cols = np.asarray(colors, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or cols.shape != pts.shape:
        raise ValueError("points and RGB colors must have matching (N, 3) shapes")
    finite = np.isfinite(pts).all(axis=1) & np.isfinite(cols).all(axis=1)
    if not finite.all():
        logger.warning("Omitting %d cloud rows with non-finite coordinates or colors", (~finite).sum())
    pts, cols = pts[finite], cols[finite]
    if np.any((cols < 0) | (cols > 255)):
        raise ValueError("RGB colors must be in the range 0-255")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Z (mm)")
    ax.set_zlabel("Y (mm)")
    if not len(pts):
        ax.set_title(f"{title}\nNo finite points to display")
        ax.set_box_aspect((1, 1, 1))
        return
    xyz = pts[:, [0, 2, 1]]
    lower = np.percentile(xyz, 1, axis=0)
    upper = np.percentile(xyz, 99, axis=0)
    spans = upper - lower
    spans[spans <= 0] = 1.0
    padding = np.maximum(spans * 0.05, float(spans.max()) * 0.02)
    lower, upper = lower - padding, upper + padding
    ax.set_xlim(lower[0], upper[0])
    ax.set_ylim(lower[1], upper[1])
    ax.set_zlim(lower[2], upper[2])
    spans = upper - lower
    ax.set_box_aspect(spans / spans.max())
    step = max(1, (len(pts) + max_points - 1) // max_points)
    sampled = xyz[::step]
    ax.scatter(*sampled.T, c=cols[::step] / 255.0, s=0.5, marker=".", depthshade=False)
    ax.set_title(f"{title}\nShowing {len(sampled):,} of {len(pts):,} finite points")


def show_disparity_heatmap(disparity: np.ndarray, save_path: str) -> None:
    """Save disparity (H, W) in px as a headless jet PNG; return None."""
    fig = Figure(figsize=(10, 6))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(disparity, cmap="jet")
    fig.colorbar(im, ax=ax, label="disparity (px)")
    ax.set_title("Disparity Map (SGBM)")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    logger.info("Saved disparity heatmap: %s", save_path)


def show_point_cloud_preview(points: np.ndarray, colors: np.ndarray, save_path: str) -> None:
    """Save points (N,3) mm with RGB colors as a headless scatter PNG; return None."""
    fig = Figure(figsize=(10, 8))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    _draw_cloud(ax, points, colors, "3D Point Cloud Preview", 50000)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    logger.info("Saved point cloud preview: %s", save_path)


def show_input_pair(left: np.ndarray, right: np.ndarray, title: str = "Stereo input") -> None:
    """Show left/right BGR photos until closed; skip without a GUI; return None."""
    if not enable_interactive():
        return
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    try:
        for ax, image, label in zip(axes, (left, right), ("Left", "Right")):
            ax.imshow(image[:, :, ::-1])
            ax.set_title(label)
            ax.set_axis_off()
        fig.suptitle(f"{title} - close this window to reconstruct")
        fig.tight_layout()
        plt.show(block=True)
    finally:
        plt.close(fig)


def show_intermediate_stages(left, right, raw, filtered, depth, title="Stereo stages") -> None:
    """Show inputs, raw/filtered disparity (px) and depth (mm) until closed; return None."""
    if not enable_interactive():
        return
    import matplotlib.pyplot as plt

    def tonemap(image):
        finite = np.isfinite(image)
        norm = np.zeros(image.shape, np.float32)
        if finite.any():
            lo, hi = np.nanpercentile(image, (1, 99))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
            norm[finite] = np.clip((image[finite] - lo) / max(hi - lo, 1e-6), 0, 1)
        return (norm * 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    try:
        axes[0, 0].imshow(left[:, :, ::-1])
        axes[0, 0].set_title("Left (rectified)")
        axes[0, 1].imshow(right[:, :, ::-1])
        axes[0, 1].set_title("Right (rectified)")
        axes[0, 2].imshow(tonemap(raw), cmap="jet", vmin=0, vmax=255)
        axes[0, 2].set_title("Raw disparity (SGBM)")
        axes[1, 0].imshow(tonemap(filtered), cmap="jet", vmin=0, vmax=255)
        axes[1, 0].set_title("Filtered disparity (LR-checked)")
        axes[1, 1].imshow(tonemap(depth), cmap="plasma", vmin=0, vmax=255)
        axes[1, 1].set_title("Depth (mm, 1-99 pct)")
        valid = np.isfinite(filtered)
        axes[1, 2].imshow(valid.astype(np.uint8) * 255, cmap="gray")
        axes[1, 2].set_title(f"Valid: {100.0 * valid.mean():.1f}%")
        for ax in axes.ravel():
            ax.set_axis_off()
        fig.suptitle(f"{title} - close to continue")
        fig.tight_layout()
        plt.show(block=True)
    finally:
        plt.close(fig)


def show_point_cloud_interactive(points, colors, title="Interactive 3D point cloud") -> None:
    """Show rotatable points (N,3) mm with RGB colors until closed; return None."""
    if not enable_interactive():
        return
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8))
    try:
        _draw_cloud(fig.add_subplot(111, projection="3d"), points, colors,
                    f"{title} - drag to rotate", 30000)
        fig.tight_layout()
        plt.show(block=True)
    finally:
        plt.close(fig)
