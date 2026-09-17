"""Configuration-driven sequential stereo reconstruction CLI."""
import argparse
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import calibration, disparity, pointcloud, stereo_calibration, utils, visualize
from src.depth import disparity_to_depth
from src.errors import CalibrationError

logger = logging.getLogger(__name__)


def run_calibrate(config: dict, output_dir: str) -> None:
    """Input YAML settings and output directory; save intrinsics and paired corners.

    Archive contains K (px), distortion, RMS (px), object points (mm), corners (px).
    Return None. Synchronized chessboard pairs must have identical filenames.
    """
    chess, settings = config["chessboard"], config["calibration"]
    pattern, square = tuple(chess["pattern_size"]), chess["square_size_mm"]
    minimum = settings["min_images"]
    left = calibration.list_images(settings["left_images_dir"])
    right = calibration.list_images(settings["right_images_dir"])
    first = calibration.calibrate_camera(left, pattern, square, min_images=minimum)
    second = calibration.calibrate_camera(right, pattern, square, min_images=minimum)
    obj, corners_l, corners_r, size = calibration.collect_stereo_points(
        left, right, pattern, square, minimum
    )
    if first["image_size"] != second["image_size"]:
        raise ValueError("Calibration camera image sizes differ")
    path = Path(output_dir) / "calibration.npz"
    np.savez(path, K_left=first["K"], dist_left=first["dist"],
             K_right=second["K"], dist_right=second["dist"],
             rms_left=first["rms_error"], rms_right=second["rms_error"],
             objpoints=np.asarray(obj), imgpoints_l=np.asarray(corners_l),
             imgpoints_r=np.asarray(corners_r), image_size=np.asarray(size))
    logger.info("Saved calibration: %s; paired corners shape=%s", path, np.shape(corners_l))


def run_reconstruct(left_path: str, right_path: str, config: dict, output_dir: str) -> None:
    """Input same-rig image paths, YAML settings, output directory; return None.

    Save rectified PNGs, disparity heatmap (px), depth.npy (mm), PLY (mm/RGB), preview.
    """
    output = Path(output_dir)
    with np.load(output / "calibration.npz", allow_pickle=False) as archive:
        required = {"K_left", "dist_left", "K_right", "dist_right", "image_size",
                    "objpoints", "imgpoints_l", "imgpoints_r"}
        if not required.issubset(archive.files):
            raise ValueError("Incomplete calibration archive; run calibrate again")
        data = {key: archive[key] for key in required}
    K1, dist1 = data["K_left"], data["dist_left"]
    K2, dist2 = data["K_right"], data["dist_right"]
    left, right = utils.load_image_color(left_path), utils.load_image_color(right_path)
    utils.validate_image_pair_shapes(left, right)
    image_size = (left.shape[1], left.shape[0])
    if image_size != tuple(data["image_size"]):
        raise ValueError(f"Input size {image_size} differs from calibration {tuple(data['image_size'])}")

    logger.info("Stage 1 rectification start: shape=%s", left.shape)
    stereo = stereo_calibration.stereo_calibrate(
        K1, dist1, K2, dist2, list(data["objpoints"]), list(data["imgpoints_l"]),
        list(data["imgpoints_r"]), image_size
    )
    maps = stereo_calibration.rectify(K1, dist1, K2, dist2, stereo["R"], stereo["T"], image_size)
    rect_left, rect_right = stereo_calibration.apply_rectification(left, right, maps)
    for name, image in (("rectified_left.png", rect_left), ("rectified_right.png", rect_right)):
        if not cv2.imwrite(str(output / name), image):
            raise IOError(f"Could not write image: {output / name}")
    logger.info("Stage 1 rectification end: shape=%s", rect_left.shape)

    gray_l = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    logger.info("Stage 2 disparity start: shape=%s", gray_l.shape)
    disp = disparity.compute_disparity(gray_l, gray_r, config["stereo"])
    visualize.show_disparity_heatmap(disp, str(output / "disparity.png"))
    logger.info("Stage 2 disparity end: shape=%s", disp.shape)

    logger.info("Stage 3 depth start: shape=%s", disp.shape)
    depth = disparity_to_depth(disp, float(K1[0, 0]), float(np.linalg.norm(stereo["T"])))
    np.save(output / "depth.npy", depth)
    logger.info("Stage 3 depth end: shape=%s", depth.shape)

    logger.info("Stage 4 point cloud start: disparity shape=%s", disp.shape)
    points_3d = pointcloud.reproject_to_3d(disp, maps["Q"])
    points, colors = pointcloud.build_point_cloud(points_3d, rect_left, disp)
    pointcloud.save_ply(str(output / "pointcloud.ply"), points, colors)
    visualize.show_point_cloud_preview(points, colors, str(output / "pointcloud_preview.png"))
    logger.info("Stage 4 point cloud end: points shape=%s colors shape=%s", points.shape, colors.shape)


def main(argv=None) -> int:
    """Input optional CLI argument strings; return exit status (0 success, 1 failure)."""
    parser = argparse.ArgumentParser(description="Stereo depth estimation and 3D reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)
    calibrate = sub.add_parser("calibrate", help="Calibrate from synchronized chessboard images")
    calibrate.add_argument("--config", required=True)
    reconstruct = sub.add_parser("reconstruct", help="Reconstruct one pair from the calibrated rig")
    reconstruct.add_argument("--left", required=True)
    reconstruct.add_argument("--right", required=True)
    reconstruct.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    utils.setup_logging()
    try:
        config = utils.load_config(args.config)
        output = Path(config["paths"]["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        if args.command == "calibrate":
            run_calibrate(config, str(output))
        else:
            run_reconstruct(args.left, args.right, config, str(output))
        return 0
    except (CalibrationError, ValueError, OSError, cv2.error, KeyError, TypeError, EOFError) as exc:
        logger.error("%s", " ".join(str(exc).splitlines()))
        return 1


if __name__ == "__main__":
    sys.exit(main())
