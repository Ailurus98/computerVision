"""Demo-only stereo reconstruction CLI (Middlebury scenes)."""
import argparse
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import disparity, pointcloud, utils, visualize

logger = logging.getLogger(__name__)


def run_scene_reconstruction(left_path, right_path, config, output_dir, intermediates=False):
    """Input rectified BGR paths, f/cx/cy/doffs (px), baseline_mm, output, steps flag.

    Output points (N,3) mm, colors RGB, stages dict; saves depth (mm), PNGs, PLY.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    settings = dict(config["stereo"])
    f, doffs, baseline = (float(settings.pop(key)) for key in ("f", "doffs", "baseline_mm"))
    left, right = utils.load_image_color(str(left_path)), utils.load_image_color(str(right_path))
    utils.validate_image_pair_shapes(left, right)
    cx, cy = float(settings.pop("cx", left.shape[1] / 2)), float(settings.pop("cy", left.shape[0] / 2))
    if not np.isfinite([f, doffs, baseline, cx, cy]).all() or f <= 0 or baseline <= 0:
        raise ValueError("Scene intrinsics must be finite with positive f and baseline_mm")
    logger.info("Stage 1 rectified grayscale start: shape=%s", left.shape)
    gray_l, gray_r = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in (left, right))
    logger.info("Stage 1 rectified grayscale end: left=%s right=%s", gray_l.shape, gray_r.shape)
    logger.info("Stage 2 filtered disparity start: shape=%s", gray_l.shape)
    raw, filtered, valid = disparity.compute_disparity_filtered(gray_l, gray_r, settings)
    # Occlusion strip: left border of width numDisparities has no right match
    # (epipolar geometry, Module 2). This removes the smeared left-edge wall.
    ndisp = settings.get("numDisparities", settings.get("num_disparities", 0))
    try:
        ndisp = int(ndisp)
    except (TypeError, ValueError):
        ndisp = 0
    if ndisp > 0 and ndisp < filtered.shape[1]:
        valid[:, :ndisp] = False
    filtered = np.where(valid, filtered, np.nan).astype(np.float32)
    logger.info("Stage 2 filtered disparity end: raw=%s filtered=%s valid=%.1f%%",
                raw.shape, filtered.shape, 100.0 * float(valid.mean()))
    logger.info("Stage 3 depth start: shape=%s", filtered.shape)
    denominator = filtered + doffs
    depth = np.full(filtered.shape, np.nan, np.float32)
    np.divide(baseline * f, denominator, out=depth, where=np.isfinite(denominator) & (denominator > 0))
    depth, cap = pointcloud.cap_depth_outliers(depth, 99.5)
    np.save(output / "depth.npy", depth)
    logger.info("Stage 3 depth end: shape=%s; saved %s", depth.shape, output / "depth.npy")
    logger.info("Stage 4 point cloud start: shape=%s", depth.shape)
    v, u = np.indices(depth.shape, dtype=np.float32)
    xyz = np.stack(((u - cx) * depth / f, (v - cy) * depth / f, depth), axis=-1)
    finite = np.isfinite(xyz).all(axis=-1)
    points, colors = xyz[finite].astype(np.float32), cv2.cvtColor(left, cv2.COLOR_BGR2RGB)[finite]
    points, colors = pointcloud.clean_point_cloud(points, colors, z_percentile=99.5)
    pointcloud.save_ply(str(output / "pointcloud.ply"), points, colors)
    visualize.show_point_cloud_preview(points, colors, str(output / "pointcloud_preview.png"))
    logger.info("Stage 4 point cloud end: points=%s colors=%s", points.shape, colors.shape)

    def save_stage(path, image, colored=False):
        """Write a min-max normalized PNG (jet if colored); return None."""
        mask = np.isfinite(image).astype(np.uint8)
        norm = cv2.normalize(np.where(mask, image, 0), None, 0, 255, cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U, mask=mask)
        if colored:
            norm = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
        if not cv2.imwrite(str(path), norm):
            raise OSError(f"Could not write image: {path}")

    save_stage(output / "disparity.png", filtered, True)
    if intermediates:
        steps = output / "steps"
        steps.mkdir(parents=True, exist_ok=True)
        for name, image in (("rectified_left", gray_l), ("rectified_right", gray_r),
                            ("disparity_raw", raw), ("disparity_filtered", filtered), ("depth", depth)):
            save_stage(steps / f"{name}.png", image, name.startswith("disparity"))
    stages = {"left": left, "right": right, "raw": raw, "filtered": filtered, "depth": depth}
    return points, colors, stages


def main(argv=None) -> int:
    """Input optional CLI argument strings; return exit status (0 success, 1 failure)."""
    parser = argparse.ArgumentParser(description="Demo-only stereo depth estimation and 3D reconstruction")
    sub = parser.add_subparsers(dest="command")
    default_config = str(Path(__file__).resolve().parents[1] / "config" / "config.yaml")
    demo = sub.add_parser("demo", help="Choose one of three portable interactive examples")
    demo.add_argument("--scene", choices=("1", "2", "3"))
    demo.add_argument("--headless", action="store_true", help="Save results without opening windows")
    demo.add_argument("--output", help="Override the portable user-cache output directory")
    demo.add_argument("--config", default=default_config)
    sub.add_parser("fetch", help="One-time download of all demo photos into the user cache")
    args = parser.parse_args(argv)
    utils.setup_logging()
    try:
        if getattr(args, "command", None) == "fetch":
            from src.demo import cache_directory, fetch_all_data

            fetch_all_data(cache_directory())
            return 0
        config = utils.load_config(getattr(args, "config", default_config))
        from src.demo import run_demo

        run_demo(config, getattr(args, "scene", None), getattr(args, "headless", False),
                 getattr(args, "output", None))
        return 0
    except (ValueError, OSError, cv2.error, KeyError, TypeError, EOFError) as exc:
        logger.error("%s", " ".join(str(exc).splitlines()))
        return 1


if __name__ == "__main__":
    sys.exit(main())
