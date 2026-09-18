# Stereo Depth Estimation and 3D Point Cloud Reconstruction from a Calibrated Binocular Camera Pair

**Student Name:** Ansuman Samal
**Registration Number:** 24BAI10965
**Course:** CSE3010 – Computer Vision
**Institution:** Vellore Institute of Technology, Bhopal
**Submission Type:** Build-Your-Own-Project (Flipped Course Evaluation)
**Repository:** https://github.com/Ailurus98/computerVision

---

## 1. Problem Statement

Human vision perceives depth largely through binocular disparity: the brain compares the slightly different images formed on the left and right retina and infers how far away each surface is. Computer Vision reproduces this with two cameras instead of two eyes.

Given two images of a static scene captured by a fixed, calibrated binocular camera pair, the problem is to **reconstruct the 3D geometry of that scene as a colored point cloud

---

## 2. Scope of the Project

### 2.1 In Scope

| Area | What is included |
|---|---|
| Camera calibration | Single-camera chessboard calibration (≥12 images per camera) producing `K` and distortion coefficients |
| Stereo calibration | Extrinsics `R`, `T`; rectification maps; reprojection matrix `Q` |
| Rectification | Warping both views so epipolar lines are horizontal and row-aligned |
| Disparity estimation | Semi-Global Block Matching (SGBM), plus a filtered variant with left–right consistency checking, median filtering, and border-occlusion masking |
| Depth computation | Conversion of disparity to metric depth in millimetres, with percentile-based outlier capping |
| 3D reconstruction | Reprojection to 3D, statistical point-cloud cleaning, and export to `.ply` |
| Visualization | Rectified-pair view, disparity heatmap, depth heatmap, and a static 3D point-cloud preview |
| Benchmark data | Middlebury 2014 stereo scenes (Motorcycle, Adirondack, Jadeplant), fetched and cached by the CLI |
| Interface | A command-line interface: `fetch` and `demo --scene {1,2,3} [--headless]` |
| Storage | Entirely file-based — `.npz` calibration cache, `.npy` depth array, `.ply` point cloud, `.png` previews |
| Testing | `pytest` unit tests plus documented qualitative validation |

### 2.2 Explicitly Out of Scope

These are deliberate boundaries, not omissions:

- **No machine learning or deep stereo networks.** Learned matching belongs to later modules.
- **No real-time or video stereo.** The pipeline processes still image pairs; temporal smoothing and optical flow are future work.
- **No multi-view or structure-from-motion.** The rig is a fixed binocular pair only.
- **No mesh generation or surface reconstruction.** The deliverable is a raw point cloud, not a watertight mesh.
- **No relational database.** All persistent state is file-based; there is therefore no ER diagram or schema design in the report.
- **No graphical user interface.** Interaction is through the CLI and matplotlib preview windows.
- **No object detection, segmentation, or scene understanding.** The output is geometry, not semantics.
- **No GPU requirement.** Everything runs on a standard CPU.

### 2.3 Assumptions and Constraints

- The scene is **static** during capture, and the two images are taken **simultaneously** or of an unchanging scene.
- The stereo rig is **rigid**: the baseline and relative orientation do not change between calibration and capture.
- Scenes are assumed to have **sufficient texture**; textureless or repetitively patterned regions will produce disparity holes, which is an expected and documented limitation rather than a defect.
- Lighting is assumed roughly consistent between the left and right views.
- `numDisparities` and `blockSize` are scene- and baseline-dependent and generally require re-tuning per stereo pair.
- A full run on a 640×480 pair should complete in under 15 seconds on a standard CPU.

---

## 3. Target Users

**Primary user — the student/developer (single actor).** The project has exactly one human role. There are no administrative, networked, or multi-user roles, which is why the use case diagram shows a single actor driving every interaction through the command line.

Beyond the immediate submission, the work is intended to be useful to:

- **Computer Vision students** studying Module 2, who need a readable, runnable reference implementation of the calibration → rectification → disparity → depth → point cloud chain, with each stage inspectable in isolation.
- **Course evaluators**, who need to verify that the implemented pipeline matches the documented design and the syllabus's indicative experiments, and can do so by running two commands against a known benchmark dataset.
- **Robotics, AR, and 3D-scanning practitioners prototyping a stereo rig**, who need a minimal, dependency-light baseline to validate their hardware and tune matching parameters before committing to a heavier framework.
- **Educators**, who can use the per-stage intermediate outputs (rectified pair, raw vs. filtered disparity, depth map, validity mask) as teaching artifacts demonstrating what each geometric step actually does.

---

## 4. High-Level Features

### 4.1 Core Pipeline Features

1. **Camera calibration** — detects 9×6 internal chessboard corners across a set of at least 12 images per camera and solves for `K` and the distortion coefficients, caching the result so calibration need not be repeated.
2. **Stereo calibration and rectification** — computes `R`, `T`, the rectification maps, and `Q`, then produces a rectified image pair in which epipolar lines are horizontal and aligned.
3. **Filtered disparity estimation** — runs SGBM in both directions and layers on:
   - a left–right consistency check to reject mismatches,
   - a 3×3 median residual filter to suppress speckle noise,
   - left-border occlusion-strip masking (`valid[:, :numDisparities] = False`).
4. **Metric depth computation** — converts disparity to depth in millimetres and caps outliers at the 1st–99th percentile range.
5. **3D point cloud reconstruction** — reprojects valid pixels to `(X, Y, Z)` using `Q`, attaches per-point RGB color from the left image, applies statistical cleaning, and writes a standard `.ply` file.

### 4.2 Usability and Tooling Features

6. **Two-command CLI** — `python src/main.py fetch` downloads and caches the Middlebury 2014 scenes; `python src/main.py demo --scene {1,2,3} [--headless]` runs the full pipeline on the chosen scene.
7. **Headless mode** — `--headless` suppresses interactive windows so the pipeline can run on a server or in CI.
8. **Benchmark dataset integration** — three selectable Middlebury scenes shipped with known-good rectified geometry and `calib.txt` intrinsics, enabling validation without a physical camera rig.
9. **Single-file configuration** — all tunable parameters exposed through one YAML config file.
10. **Visualization suite** — rectified-pair view, disparity heatmap (`jet`), depth heatmap (`inferno`), validity mask, and a static matplotlib 3D point-cloud preview that avoids making Open3D a hard dependency.
11. **Open output formats** — `.ply` point clouds readable by MeshLab, CloudCompare, and Open3D without extra tooling.

### 4.3 Engineering Quality Features

12. **Single-responsibility module layout** — ten Python modules in `src/`, each under roughly 200 lines, written in a procedural function-based style with documented public functions.
13. **Descriptive error handling** — custom exception types (e.g. `CalibrationError`) raised on missing images, mismatched left/right dimensions, or failed chessboard detection, instead of unhandled exceptions.
14. **Structured logging** — progress at `INFO` and failures at `ERROR` via Python's `logging` module, not `print` statements.
15. **Automated tests** — `pytest` coverage of corner detection, disparity map shape and parameter validation, and point-cloud integrity (no NaN/Inf, matching point and color counts).
16. **CPU-only operation** — no GPU required anywhere in the pipeline.

---

## 5. Module Map

| Module | Exported functions |
|---|---|
| `calibration.py` | `calibrate_camera()` |
| `stereo_calibration.py` | `stereo_calibrate()`, `rectify()` |
| `disparity.py` | `compute_disparity()`, `compute_disparity_filtered()` |
| `depth.py` | `disparity_to_depth()`, `cap_depth_outliers()` |
| `pointcloud.py` | `reproject_to_3d()`, `clean_point_cloud()`, `save_ply()` |
| `visualize.py` | `show_disparity_heatmap()`, `show_point_cloud_preview()` |
| `demo.py` | `fetch_dataset()`, `run_demo(scene)` |
| `main.py` | CLI entry point and pipeline orchestration |
| `utils.py` | Shared I/O and array helpers |
| `errors.py` | Custom exception classes |

---

## 6. Success Criteria

The project is considered successful when:

- Rectified epipolar lines are visibly horizontal across the stereo pair.
- The filtered disparity map is smooth over textured regions, with holes confined to genuinely occluded or textureless areas.
- The exported point cloud recognizably reproduces the scene's geometry when viewed from a rotated angle.
- The full `pytest` suite passes.
- Optionally, RMSE against Middlebury ground-truth disparity is computed over valid, non-occluded pixels.
