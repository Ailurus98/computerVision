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
SCENES = {"1": ("Motorcycle (vehicle)", "Motorcycle-perfect"),
          "2": ("Adirondack (outdoor backyard)", "Adirondack-perfect"),
          "3": ("Jadeplant (plant)", "Jadeplant-perfect")}
MIRRORS = {
    "Motorcycle-perfect": ("https://raw.githubusercontent.com/zaitera/Stereo-Vision/master/data/Middlebury/Motorcycle-perfect/", ("imL.png", "imR.png")),
    "Adirondack-perfect": ("https://raw.githubusercontent.com/jiafeng5513/Evision/master/data/Adirondack-perfect/", ("im0.png", "im1.png")),
    "Jadeplant-perfect": ("https://raw.githubusercontent.com/zaitera/Stereo-Vision/master/data/Middlebury/Jadeplant-perfect/", ("imL.png", "imR.png")),
}
CALIB_URLS = {
    "Motorcycle-perfect": ["https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Motorcycle-perfect/calib.txt"],
    "Adirondack-perfect": ["https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Adirondack-perfect/calib.txt",
                            "https://raw.githubusercontent.com/jiafeng5513/Evision/master/data/Adirondack-perfect/calib.txt"],
    "Jadeplant-perfect": ["https://vision.middlebury.edu/stereo/data/scenes2014/datasets/Jadeplant-perfect/calib.txt"],
}
INDICES = tuple(range(1, 10)) + tuple(range(11, 15))

def cache_directory() -> Path:
    """Input: environment; output: portable cache path; no physical units."""
    base = os.environ.get("LOCALAPPDATA") if os.name == "nt" else os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "stereo-reconstruction"

def _valid_jpg(path: Path) -> bool:
    """Input image path; output True only for a decodable 640x480 color JPG."""
    if not (path.is_file() and 0 < path.stat().st_size <= 5_000_000):
        return False
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image is not None and image.shape == (480, 640, 3)


def prepare_chessboards(root: Path, allow_download: bool = False) -> tuple[Path, Path]:
    """Input cache root and download flag; output validated left/right 640x480 JPG folders.

    With allow_download False (default) missing files raise OSError pointing at
    the fetch command instead of touching the network.
    """
    folders = root / "calibration" / "left", root / "calibration" / "right"
    missing = False
    for side, folder in zip(("left", "right"), folders):
        folder.mkdir(parents=True, exist_ok=True)
        for index in INDICES:
            path = folder / f"{index:02d}.jpg"
            if _valid_jpg(path):
                continue
            url = f"{SOURCE}/{side}{index:02d}.jpg"
            if not allow_download:
                missing = True
                continue
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    content = response.read(5_000_001)
            except (urllib.error.URLError, TimeoutError) as exc:
                raise OSError(f"Sample download failed: {url}; check internet and retry: {exc}") from exc
            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if len(content) > 5_000_000 or image is None or image.shape != (480, 640, 3):
                raise ValueError(f"Invalid reference image response: {url}")
            path.write_bytes(content)
    if missing:
        raise OSError("Chessboard photos not available offline. Run once: python src/main.py fetch")
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

def _fetch(url: str, limit: int) -> bytes:
    """Input URL and byte limit; output response bytes; network errors become OSError."""
    logger.info("Downloading scene data: %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read(limit + 1)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise OSError(f"Sample download failed: {url}; check internet and retry: {exc}") from exc


def _valid_scene_png(path: Path) -> bool:
    """Input image path; output True only for a decodable color PNG >=1000 px per axis."""
    if not (path.is_file() and 0 < path.stat().st_size <= 30_000_000):
        return False
    try:
        content = path.read_bytes()
    except OSError:
        return False
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
    return (content.startswith(b"\x89PNG\r\n\x1a\n") and image is not None and image.ndim == 3
            and image.shape[2] >= 3 and min(image.shape[:2]) >= 1000)


def _valid_scene_folder(folder: Path) -> bool:
    """Input scene folder; output True only for two valid PNGs plus a parseable calib.txt."""
    try:
        return (_valid_scene_png(folder / "im0.png") and _valid_scene_png(folder / "im1.png")
                and bool(parse_calib(folder / "calib.txt")))
    except (OSError, ValueError):
        return False


def prepare_scenes(root: Path, allow_download: bool = False) -> dict[str, Path]:
    """Input cache root and download flag; output validated scene-name folders.

    Resolution per scene: repository data/scenes/<name>/ first (offline forks that
    vendor the images), then the user cache. With allow_download False (default)
    nothing is fetched; missing data raises OSError pointing at the fetch command.
    """
    bundled = Path(__file__).resolve().parents[1] / "data" / "scenes"
    folders = {}
    for _, name in SCENES.values():
        candidate = bundled / name
        folder = candidate if _valid_scene_folder(candidate) else root / "scenes" / name
        folders[name] = folder
    for name, folder in folders.items():
        folder.mkdir(parents=True, exist_ok=True)
        if _valid_scene_folder(folder):
            continue
        if not allow_download:
            raise OSError(f"Scene '{name}' not available offline. Run once: python src/main.py fetch")
        mirror, (remote_left, remote_right) = MIRRORS[name]
        for filename, remote in (("im0.png", remote_left), ("im1.png", remote_right)):
            path = folder / filename
            content = path.read_bytes() if _valid_scene_png(path) else _fetch(mirror + remote, 30_000_000)
            image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_UNCHANGED)
            if (not content.startswith(b"\x89PNG\r\n\x1a\n") or image is None or image.ndim != 3
                    or image.shape[2] < 3 or min(image.shape[:2]) < 1000):
                raise ValueError(f"Invalid scene PNG: {path}")
            path.write_bytes(content)
        path = folder / "calib.txt"
        if not (path.is_file() and path.stat().st_size < 10_000):
            failures = []
            for url in CALIB_URLS[name]:
                try:
                    content = _fetch(url, 10_000)
                    break
                except OSError as exc:
                    failures.append(str(exc))
            else:
                raise OSError(f"Scene calibration download failed: {'; '.join(failures)}")
            path.write_bytes(content)
        parse_calib(path)
    return folders


def fetch_all_data(root: Path) -> None:
    """Input cache root; download all demo scenes and chessboards; return None.

    This is the only function allowed to touch the network. Run it once via
    `python src/main.py fetch`; every other entry point works fully offline.
    """
    prepare_scenes(root, allow_download=True)
    prepare_chessboards(root, allow_download=True)
    logger.info("All demo data cached under %s", root)

def run_demo(config, scene=None, headless=False, output_dir=None):
    """Input: config, scene 1..3, headless flag, output path; output: None, exports XYZ/depth in mm."""
    from src.main import run_calibrate, run_scene_reconstruction
    if scene is None:
        for label, (title, _) in SCENES.items():
            logger.info("%s: %s", label, title)
        scene = input("Select scene 1, 2, or 3: ").strip()
    if scene not in SCENES:
        raise ValueError("Select scene 1, 2, or 3")
    title, name = SCENES[scene]
    root = cache_directory()
    folder = prepare_scenes(root)[name]
    settings = {key: dict(value) for key, value in config.items()}
    calib = parse_calib(folder / "calib.txt")
    # Syllabus-only tuning: block 5->7 = larger Module-1 aggregation window
    # (smoother, fewer speckles); LR + median checks = Module-2 matching test.
    settings["stereo"] = dict(num_disparities={"1": 272, "2": 288, "3": 640}[scene], block_size=7,
                              lr_max_diff=1.0, median_max_diff=2.0,
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
    result = run_scene_reconstruction(left_path, right_path, settings, str(output), intermediates=True)
    points, colors = result[0], result[1]
    stages = result[2] if len(result) > 2 else None
    if interactive and stages is not None:
        visualize.show_intermediate_stages(stages["left"], stages["right"], stages["raw"],
                                           stages["filtered"], stages["depth"], title)
    if interactive:
        visualize.show_point_cloud_interactive(points, colors, title)
