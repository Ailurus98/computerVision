"""Portable three-scene demonstration and first-run sample-data setup."""
import logging
import os
from pathlib import Path
import urllib.error
import urllib.request

import cv2
import numpy as np

from src import utils, visualize

logger = logging.getLogger(__name__)
SOURCE = "https://raw.githubusercontent.com/opencv/opencv/5.x/samples/data"
SCENES = {"1": ("Scene 1 - chessboard view 01", "01"),
          "2": ("Scene 2 - chessboard view 03", "03"),
          "3": ("Scene 3 - chessboard view 06", "06")}
INDICES = tuple(range(1, 10)) + tuple(range(11, 15))


def cache_directory() -> Path:
    """Return a portable user-writable cache directory; no physical units."""
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "stereo-reconstruction"


def prepare_reference_data(root: Path) -> tuple[Path, Path]:
    """Input cache root; download missing OpenCV JPG pairs and return camera folders.

    Downloads only during demo setup; subsequent runs reuse validated 640x480 files.
    The reconstruction pipeline itself remains offline. No scripts are downloaded.
    """
    folders = root / "calibration" / "left", root / "calibration" / "right"
    for side, folder in zip(("left", "right"), folders):
        folder.mkdir(parents=True, exist_ok=True)
        for index in INDICES:
            path = folder / f"{index:02d}.jpg"
            if path.is_file():
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is not None and image.shape == (480, 640, 3):
                    continue
            url = f"{SOURCE}/{side}{index:02d}.jpg"
            logger.info("Downloading reference photo: %s", url)
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    content = response.read(5_000_001)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise OSError(f"Sample download failed: {url}; check internet and retry: {exc}") from exc
            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if len(content) > 5_000_000 or image is None or image.shape != (480, 640, 3):
                raise ValueError(f"Invalid reference image response: {url}")
            path.write_bytes(content)
    return folders


def run_demo(config, scene=None, headless=False, output_dir=None):
    """Input settings, optional scene 1..3, headless flag and output path; return None.

    Show BGR input photos, calibrate in mm, export a cloud, then open interactive 3D.
    Samples use an assumed square size: absolute millimeter accuracy is not verified.
    """
    from src.main import run_calibrate, run_reconstruct

    if scene is None:
        for label, (title, _) in SCENES.items():
            logger.info("%s: %s", label, title)
        logger.info("Select scene 1, 2, or 3, then press Enter:")
        scene = input().strip()
    if scene not in SCENES:
        raise ValueError("Select scene 1, 2, or 3")
    title, index = SCENES[scene]
    root = cache_directory()
    left_dir, right_dir = prepare_reference_data(root)
    settings = {key: dict(value) for key, value in config.items()}
    settings["chessboard"]["pattern_size"] = [9, 6]
    settings["calibration"].update(left_images_dir=str(left_dir), right_images_dir=str(right_dir))
    output = Path(output_dir).expanduser().resolve() if output_dir else root / "output" / f"scene{scene}"
    output.mkdir(parents=True, exist_ok=True)
    left_path, right_path = left_dir / f"{index}.jpg", right_dir / f"{index}.jpg"
    left, right = utils.load_image_color(str(left_path)), utils.load_image_color(str(right_path))
    interactive = not headless and visualize.enable_interactive()
    logger.info("%s; square size %.2f mm is assumed, not verified", title,
                settings["chessboard"]["square_size_mm"])
    if interactive:
        logger.info("Close the input-photo window to start reconstruction")
        visualize.show_input_pair(left, right, title)
    run_calibrate(settings, str(output))
    points, colors = run_reconstruct(str(left_path), str(right_path), settings, str(output))
    logger.info("Results saved to %s", output)
    if interactive:
        logger.info("Drag the 3D view to rotate; use the toolbar to zoom; close to exit")
        visualize.show_point_cloud_interactive(points, colors, title)
