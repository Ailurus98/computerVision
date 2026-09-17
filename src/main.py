"""CLI orchestration: calibrate and reconstruct subcommands."""
import argparse
import logging
import os
import sys

import numpy as np

from src import calibration, disparity, pointcloud, stereo_calibration, utils, visualize
from src.depth import disparity_to_depth
from src.errors import CalibrationError

logger = logging.getLogger(__name__)


def run_calibrate(config: dict, output_dir: str) -> None:
    """Calibrate both cameras from config paths and save calibration.npz.

    Inputs: parsed config dict, output directory.
    Outputs: None; writes output/calibration.npz.
    """
    chess = config["chessboard"]
    calib_cfg = config["calibration"]
    min_images = int(calib_cfg.get("min_images", 12))
    pattern = tuple(chess["pattern_size"])
    square = float(chess["square_size_mm"])

    results = {}
    for side, key in (("left", "left_images_dir"), ("right", "right_images_dir")):
        d = calib_cfg[key]
        paths = calibration.list_images(d)
        logger.info("Camera '%s': %d candidate images in %s", side, len(paths), d)
        if len(paths) < min_images:
            raise CalibrationError(
                f"Camera '{side}': only {len(paths)} images in {d}; need at least {min_images}."
            )
        results[side] = calibration.calibrate_camera(paths, pattern, square)
        logger.info("Camera '%s' RMS: %.4f px", side, results[side]["rms_error"])

    path = os.path.join(output_dir, "calibration.npz")
    np.savez(
        path,
        K_left=results["left"]["K"], dist_left=results["left"]["dist"],
        K_right=results["right"]["K"], dist_right=results["right"]["dist"],
        rms_left=results["left"]["rms_error"], rms_right=results["right"]["rms_error"],
        image_size=np.array(results["left"]["image_size"]),
    )
    logger.info("Saved calibration to %s", path)


def run_reconstruct(left_path: str, right_path: str, config: dict, output_dir: str) -> None:
    """Run the full stereo pipeline: rectify -> disparity -> depth -> point cloud.

    Inputs: stereo pair paths, parsed config dict, output directory.
    Outputs: None; writes rectified_left.png, rectified_right.png, disparity.png,
        depth.npy, pointcloud.ply, pointcloud_preview.png into output_dir.
    """
    calib_path = os.path.join(output_dir, "calibration.npz")
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(
            f"Calibration file not found: {calib_path} (run 'calibrate' first)"
        )
    data = np.load(calib_path)
    K1, dist1 = data["K_left"], data["dist_left"]
    K2, dist2 = data["K_right"], data["dist_right"]

    img_left = utils.load_image_color(left_path)
    img_right = utils.load_image_color(right_path)
    utils.validate_image_pair_shapes(img_left, img_right)
    image_size = (img_left.shape[1], img_left.shape[0])
    logger.info("Stereo pair loaded: shape=%s", img_left.shape)

    gray_left = utils.load_image_gray(left_path)
    gray_right = utils.load_image_gray(right_path)

    stereo = stereo_calibration.stereo_calibrate(
        K1, dist1, K2, dist2, None, None, None, image_size
    ) if False else None  # placeholder removed below

    maps = stereo_calibration.rectify(K1, dist1, K2, dist2, data["R"], data["T"], image_size) if "R" in data else None
    if maps is None:
        raise ValueError(
            "calibration.npz lacks stereo extrinsics; re-run calibration with stereo pair support"
        )

    rect_left, rect_right = stereo_calibration.apply_rectification(img_left, img_right, maps)
    rect_gray_l, rect_gray_r = stereo_calibration.apply_rectification(gray_left, gray_right, maps)
    stereo_calibration.save_rectified_pair(rect_left, rect_right, output_dir) if hasattr(stereo_calibration, "save_rectified_pair") else None

    import cv2
    cv2.imwrite(os.path.join(output_dir, "rectified_left.png"), rect_left)
    cv2.imwrite(os.path.join(output_dir, "rectified_right.png"), rect_right)

    st_cfg = config.get("stereo", {})
    disp = disparity.compute_disparity(rect_gray_l, rect_gray_r, dict(st_cfg))
    cv2.imwrite(os.path.join(output_dir, "disparity.png"), cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    visualize.show_disparity_heatmap(disp, os.path.join(output_dir, "disparity_heatmap.png"))

    focal_px = float(K1[0, 0])
    baseline_mm = float(np.linalg.norm(data["T"]))
    depth = disparity_to_depth(disp, focal_px, baseline_mm)
    np.save(os.path.join(output_dir, "depth.npy"), depth)
    logger.info("Saved depth map: %s", os.path.join(output_dir, "depth.npy"))

    points_3d = pointcloud.reproject_to_3d(disp, maps["Q"])
    points, colors = pointcloud.build_point_cloud(points_3d, rect_left, disp)
    pointcloud.save_ply(os.path.join(output_dir, "pointcloud.ply"), points, colors)
    visualize.show_point_cloud_preview(points, colors, os.path.join(output_dir, "pointcloud_preview.png"))
    logger.info("Reconstruction complete; outputs in %s", output_dir)


def main(argv=None) -> int:
    """CLI entry point.

    Inputs: optional argv list (defaults to sys.argv[1:]).
    Outputs: process exit code (0 success, 1 handled failure).
    """
    parser = argparse.ArgumentParser(description="Stereo depth estimation and 3D reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cal = sub.add_parser("calibrate", help="Calibrate both cameras from chessboard images")
    p_cal.add_argument("--config", required=True, help="Path to config.yaml")

    p_rec = sub.add_parser("reconstruct", help="Run stereo pipeline on one image pair")
    p_rec.add_argument("--left", required=True, help="Left image path")
    p_rec.add_argument("--right", required=True, help="Right image path")
    p_rec.add_argument("--config", required=True, help="Path to config.yaml")

    args = parser.parse_args(argv)
    try:
        config = utils.load_config(args.config)
        utils.setup_logging(config.get("logging", {}).get("level", "INFO") if isinstance(config.get("logging"), dict) else "INFO")
        output_dir = config.get("paths", {}).get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        if args.command == "calibrate":
            run_calibrate(config, output_dir)
        else:
            run_reconstruct(args.left, args.right, config, output_dir)
        return 0
    except (CalibrationError, ValueError, FileNotFoundError, IOError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
