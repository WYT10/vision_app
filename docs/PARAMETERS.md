# Parameters

## Core
- `mode = probe | calibrate | deploy`
- `probe_task = list | live | snap | bench`
- `roi_mode = fixed | dynamic_red_stacked`

## Warp
- `warp_width`, `warp_height`
- `target_tag_px`
- `warp_center_x_ratio`
- `warp_center_y_ratio`

`warp_center_y_ratio < 0.5` places the tag higher in the warp canvas, leaving more room below and reducing lower-edge clipping.

## Dynamic stacked trigger
- `dyn_search_x0`, `dyn_search_x1`
- `dyn_upper_y0`, `dyn_upper_y1`
- `dyn_lower_y0`, `dyn_lower_y1`
- `dyn_zone_min_pixels`
- `dyn_zone_min_ratio`
- `dyn_center_x_max_diff`
- `dyn_stable_frames_required`
- `dyn_roi_width`, `dyn_roi_height`
- `dyn_roi_gap_above_upper_zone`

### Dynamic ROI formula

- `roi_bottom = upper_y0 - roi_gap_above_upper_zone`
- `roi_top = roi_bottom - roi_height`
- `roi_left = x_center - roi_width/2`

Then clamp to image bounds.
