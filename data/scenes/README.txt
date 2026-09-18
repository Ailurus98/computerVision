Demo scene photos — stored offline, never fetched at run time
=============================================================

The three demo scenes are Middlebury 2014 calibrated stereo pairs
(Motorcycle, Adirondack, Jadeplant) plus each scene's calib.txt.

Resolution order used by the program (first valid hit wins):
1. data/scenes/<name>/ inside this repository, if a fork vendor the images
   (im0.png, im1.png, calib.txt) — fully offline, preferred.
2. The per-user cache: %LOCALAPPDATA%\stereo-reconstruction\scenes (Windows)
   or ~/.cache/stereo-reconstruction/scenes (Linux/macOS).

`python src/main.py` / `demo` NEVER downloads. If a scene is missing you get:
  "Scene '<name>' not available offline. Run once: python src/main.py fetch"
The one-time `fetch` command populates the cache (needs internet once);
every later run works offline.

Sources (Middlebury 2014 "-perfect" pairs, used with attribution):
- Motorcycle: https://github.com/zaitera/Stereo-Vision (data/Middlebury/Motorcycle-perfect/imL.png, imR.png)
- Adirondack: https://github.com/jiafeng5513/Evision (data/Adirondack-perfect/im0.png, im1.png)
- Jadeplant:  https://github.com/zaitera/Stereo-Vision (data/Middlebury/Jadeplant-perfect/imL.png, imR.png)
- calib.txt:  https://vision.middlebury.edu/stereo/data/scenes2014/ (direct download works)
- Method citation: Scharstein et al., "High-Resolution Stereo Datasets with
  Subpixel-Accurate Ground Truth", GCPR 2014.

NOTE: data/scenes/Motorcycle-perfect/im0.png is a leftover TEXT placeholder
from an earlier setup attempt — it is not a real image, will fail validation,
and is ignored by the program. It cannot be deleted from this particular
checkout (OS denies it); on a normal disk, delete it.
