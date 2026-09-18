"""Middlebury 2014 examples: https://vision.middlebury.edu/stereo/data/scenes2014/.
Inputs: scene/config paths; outputs: clouds in mm. Scene depth ignores square_size.
"""
import logging
import os
from pathlib import Path
import urllib.error
import urllib.request
import cv2
import numpy as np
from src import utils, visualize
from src.errors import CalibrationError

logger = logging.getLogger(__name__)
SOURCE = "https://raw.githubusercontent.com/opencv/opencv/5.x/samples/data"
SCENES = {"1": ("Motorcycle (vehicle)", "Motorcycle-perfect", "2860700339"),
          "2": ("Adirondack (outdoor backyard)", "Adirondack-perfect", "4152648054"),
          "3": ("Jadeplant (plant)", "Jadeplant-perfect", "4198248708")}
INDICES = tuple(range(1, 10)) + tuple(range(11, 15))

def cache_directory() -> Path:
    """Input: environment; output: portable cache path; no physical units."""
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "stereo-reconstruction"

def prepare_chessboards(root: Path) -> tuple[Path, Path]:
    """Input: cache root; output: validated left/right folders of 640x480 px JPGs."""
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

def parse_calib(calib_path) -> dict:
    """Input: key=value calibration path; output: f/cx/cy/doffs in px, baseline in mm, sizes in px."""
    data = dict(line.split("=", 1) for line in Path(calib_path).read_text(encoding="utf-8").splitlines() if "=" in line)
    data = {key.strip(): value.strip() for key, value in data.items()}
    try:
        cam = [float(token) for token in data["cam0"].strip("[]").replace(";", " ").split()]
        return dict(f=cam[0], cx=cam[2], cy=cam[5], doffs=float(data["doffs"]),
                    baseline=float(data["baseline"]), **{key: int(data[key]) for key in ("ndisp", "width", "height")})
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Invalid scene calibration: {calib_path}") from exc

def prepare_scenes(root: Path) -> dict[str, Path]:
    """Input: cache root; output: scene-name folders; PNGs >=1000 px per axis, files <=30 MB."""
    folders = {name: root / "scenes" / name for _, name, _ in SCENES.values()}
    for name, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        for filename in ("im0.png", "im1.png", "calib.txt"):
            path = folder / filename
            cached = path.is_file() and path.stat().st_size <= 30_000_000
            if cached:
                content = path.read_bytes()
            else:
                url = f"https://vision.middlebury.edu/stereo/data/scenes2014/datasets/{name}/{filename}"
                logger.info("Downloading scene data: %s", url)
                with urllib.request.urlopen(url, timeout=30) as response:
                    content = response.read(30_000_001)
            if not content or len(content) > 30_000_000:
                raise ValueError(f"Invalid scene file size: {path}")
            if filename.endswith(".png"):
                image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
                if (not content.startswith(b"\x89PNG\r\n\x1a\n") or image is None or image.ndim != 3
                        or image.shape[2] < 3 or min(image.shape[:2]) < 1000):
                    raise ValueError(f"Invalid scene PNG: {path}")
            if not cached:
                path.write_bytes(content)
            if filename == "calib.txt":
                parse_calib(path)
    return folders

def run_demo(config, scene=None, headless=False, output_dir=None):
    """Input: config, scene 1..3, headless flag, output path; output: None, exports XYZ/depth in mm."""
    from src.main import run_calibrate, run_scene_reconstruction
    if scene is None:
        for label, (title, _, _) in SCENES.items():
            logger.info("%s: %s", label, title)
        scene = input("Select scene 1, 2, or 3: ").strip()
    if scene not in SCENES:
        raise ValueError("Select scene 1, 2, or 3")
    title, name, _ = SCENES[scene]
    root = cache_directory()
    folder = prepare_scenes(root)[name]
    settings = {key: dict(value) for key, value in config.items()}
    calib = parse_calib(folder / "calib.txt")
    settings["stereo"] = dict(num_disparities={"1": 272, "2": 288, "3": 640}[scene], block_size=5,
                              f=calib["f"], doffs=calib["doffs"], baseline_mm=calib["baseline"], cx=calib["cx"], cy=calib["cy"])
    output = Path(output_dir).expanduser().resolve() if output_dir else root / "output" / f"scene{scene}"
    output.mkdir(parents=True, exist_ok=True)
    left_path, right_path = str(folder / "im0.png"), str(folder / "im1.png")
    left, right = utils.load_image_color(left_path), utils.load_image_color(right_path)
    interactive = not headless and visualize.enable_interactive()
    if interactive:
        visualize.show_input_pair(left, right, title)
    if settings.get("calibration") and not (output / "calibration.npz").exists():
        try:
            if all((root / "calibration" / side).is_dir() for side in ("left", "right")):
                left_dir, right_dir = prepare_chessboards(root)
                settings["calibration"].update(left_images_dir=str(left_dir), right_images_dir=str(right_dir))
                settings["chessboard"]["pattern_size"] = [9, 6]
            if all(Path(settings["calibration"][key]).is_dir() for key in ("left_images_dir", "right_images_dir")):
                run_calibrate(settings, str(output))
            else:
                logger.warning("Skipping optional chessboard calibration: missing directories")
        except (OSError, ValueError, CalibrationError) as exc:
            logger.warning("Skipping optional chessboard calibration: %s", exc)
    points, colors = run_scene_reconstruction(left_path, right_path, settings, str(output), intermediates=True)
    if interactive:
        visualize.show_point_cloud_interactive(points, colors, title)
