# Fixed ROI vs dynamic red-center-x

This pack now contains **both** runtime paths.

## Mode A — `fixed`

Behavior:
- crop saved `red_roi`
- crop saved `image_roi`
- compute `red_ratio` inside fixed `red_roi`
- downstream image crop is still the fixed saved `image_roi`

Use this as the baseline.

---

## Mode B — `dynamic-red-x`

Behavior:
1. apply saved warp
2. crop a fixed red search band `[x0:x1, y0:y1]`
3. HSV threshold red
4. morphology open + close
5. find the best red contour by area
6. compute contour **center x only**
7. smooth x with `red_center_alpha`
8. if red is missing briefly, hold last x for `red_miss_tolerance`
9. otherwise fallback to `red_fallback_center_x` or search-band center
10. build final image crop from:
   - `x_center`
   - `red_band_y0`
   - `roi_gap_above_band`
   - `roi_width`
   - `roi_height`
11. clamp crop to warped-image bounds
12. pass crop downstream

---

## What you can see in UI now

### `fixed`
- red fixed rectangle
- image fixed rectangle

### `dynamic-red-x`
- orange red search band
- green live `x_center` line
- blue auto image ROI
- optional `vision_app_red_mask` window

You can toggle mode at runtime in calibrate/deploy with:
- `m`

---

## Parameters

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
red_show_mask_window=1
```

Notes:
- `red_search_x1=-1` means full width
- `roi_anchor_y=-1` means use the gap-based rule
- top of dynamic image ROI = `red_band_y0 - roi_gap_above_band - roi_height`
- `red_fallback_center_x=-1` means use search-band center
- all of these are in **warped-image pixels**

---

## Why keep both modes

You need both because they answer different questions.

### `fixed`
- best baseline
- easiest to reason about
- good for repeatability

### `dynamic-red-x`
- matches your intended mechanism
- allows left-right adaptation from red location
- keeps vertical geometry deterministic

This lets you compare:
- stability
- latency
- miss behavior
- downstream model accuracy
