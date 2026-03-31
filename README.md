# vision_app compact operator pack v2

This pack now exposes **two explicit ROI runtime modes** so you can compare the old method and the new red-center method cleanly.

## ROI runtime modes

### 1) `fixed`
Classic behavior.

- save `red_roi`
- save `image_roi`
- deploy crops exactly those two fixed rectangles
- red logic reports `red_ratio` inside the fixed `red_roi`

### 2) `dynamic-red-x`
New behavior.

- define a fixed red search band in the warped image
- threshold red only inside that band
- find the best red blob
- use **blob center x only**
- build `image_roi` automatically from:
  - `x_center`
  - `red_band_y0`
  - `roi_gap_above_band`
  - `roi_width`
  - `roi_height`
- if red disappears briefly, hold last center for `red_miss_tolerance`
- if nothing valid is available, fallback to band center or `red_fallback_center_x`

This keeps the two methods separate and directly comparable.

---

## What changed in v2

- added `--roi-mode fixed|dynamic-red-x`
- added dynamic red-center runtime path in `roi_helper.cpp`
- calibrate/deploy overlays now show the active mode
- added `m` key to toggle fixed <-> dynamic-red-x in UI sessions
- optional `vision_app_red_mask` window for dynamic mode
- `rois.yml` can now also store:
  - `roi_mode`
  - dynamic red-band / crop parameters

---

## Build on Raspberry Pi

```bash
export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH

cd ~/Desktop/vision_app
rm -rf build
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

---

## Short commands

### Probe list

```bash
./vision_app --mode probe --probe-task list --device /dev/video0
```

### Probe bench

```bash
./vision_app --mode probe --probe-task bench --device /dev/video0 --headless --duration 10 --save-report ../report/probe_bench.md
```

### Calibrate old fixed mode

```bash
./vision_app --config configs/pi5_fast.conf --mode calibrate --roi-mode fixed
```

### Calibrate dynamic red-center mode

```bash
./vision_app --config configs/pi5_fast.conf --mode calibrate --roi-mode dynamic-red-x --red-show-mask-window 1
```

### Deploy old fixed mode

```bash
./vision_app --config configs/pi5_fast.conf --mode deploy --roi-mode fixed
```

### Deploy dynamic red-center mode

```bash
./vision_app --config configs/pi5_fast.conf --mode deploy --roi-mode dynamic-red-x
```

---

## Dynamic red-center parameters

These are in warped-image pixel coordinates.

```ini
roi_mode=dynamic-red-x
red_band_y0=120
red_band_y1=180
red_search_x0=0
red_search_x1=-1
roi_gap_above_band=0
roi_anchor_y=-1
roi_width=96
roi_height=96
red_min_area=40
red_max_area=0
red_morph_k=3
red_center_alpha=0.70
red_miss_tolerance=5
red_fallback_center_x=-1
red_show_mask_window=0
```

Meaning:

- `red_band_y0 / red_band_y1`: vertical band for red search
- `red_search_x0 / red_search_x1`: optional horizontal search limits
- `roi_gap_above_band`: distance between image-roi bottom and red-band top
- `roi_anchor_y`: legacy direct top-y override; keep `-1` to use the gap model
- `roi_width / roi_height`: final crop size
- `red_min_area / red_max_area`: blob filter
- `red_morph_k`: morphology kernel size
- `red_center_alpha`: smoothing weight for new x detections
- `red_miss_tolerance`: frames to hold last x before fallback
- `red_fallback_center_x`: explicit fallback x, or `-1` for band center

---

## UI behavior

### In `fixed`
You will see:
- fixed `red_roi`
- fixed `image_roi`

### In `dynamic-red-x`
You will see:
- orange search band
- green `x_center` line
- blue auto-placed `dynamic_image_roi`
- optional red-mask debug window

### In calibrate UI
- `m` toggles mode
- fixed mode keeps the old rectangle editor
- dynamic mode is live-tunable:
  - `w/s` move band
  - `i/k` band height
  - `a/d` ROI width
  - `z/x` ROI height
  - `j/l` gap above band
  - `[/]` dynamic pixel step

---

## Best workflow

1. lock warp in `calibrate`
2. compare `fixed` and `dynamic-red-x` on the same saved warp
3. save crops from both modes
4. run deploy with the same camera mode and model backend
5. compare stability, missed detections, and model accuracy

---

## File map

- `src/main.cpp` — CLI entrypoint
- `src/camera.hpp` — probe / capture helpers
- `src/calibrate.hpp` — tag detection, warp package, ROI geometry, roi yaml
- `src/deploy.hpp` — calibrate and deploy app loops, active mode overlays
- `src/roi_helper.cpp` — fixed ROI runtime + dynamic red-center runtime
- `src/model.cpp` / `src/model.hpp` — classifier runtime facade
- `docs/FUNCTIONS.md` — function inventory
- `tools/` — dataset + training + export helpers
