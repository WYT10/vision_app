# Parameter reference

## Camera

- `device`
- `width`
- `height`
- `fps`
- `fourcc`
- `buffer_size`
- `latest_only`
- `drain_grabs`

## UI

- `ui`
- `draw_overlay`
- `camera_preview_max`
- `warp_preview_max`
- `text_console`
- `text_console_font_scale`
- `text_console_padding`

## Tag / warp

- `tag_family`
- `target_id`
- `require_target_id`
- `manual_lock_only`
- `lock_frames`
- `warp_width`
- `warp_height`
- `target_tag_px`
- `warp_center_x_ratio`
- `warp_center_y_ratio`

Rule:
- `warp_center_y_ratio < 0.5` leaves more room below the tag in the warped canvas.

## Fixed ROI mode

- `roi_mode = fixed`
- `fixed_red_roi = x,y,w,h`
- `fixed_image_roi = x,y,w,h`

## Dynamic stacked mode

- `roi_mode = dynamic_red_stacked`
- `search_x0`
- `search_x1`
- `upper_y0`
- `upper_y1`
- `lower_y0`
- `lower_y1`
- `roi_width`
- `roi_height`
- `roi_gap_above_upper_zone`
- `x_smoothing_alpha`
- `miss_tolerance_frames`

Interpretation:
- upper zone is the first trigger band
- lower zone is the second trigger band
- the ROI is synthesized above the upper zone

## HSV red threshold

- `red_h1_low`
- `red_h1_high`
- `red_h2_low`
- `red_h2_high`
- `red_s_min`
- `red_v_min`

## Morphology

- `red_morph_open_k`
- `red_morph_close_k`

## Red mass gates

Per-zone gates:
- `zone_min_pixels`
- `zone_min_ratio`
- `zone_min_blob_area`
- `zone_max_blob_area`

Optional full-band gates:
- `band_min_pixels`
- `band_min_ratio`

Cross-zone consistency:
- `center_x_max_diff`
- `trigger_consecutive_frames`

## Suggested first values

```ini
search_x0=0
search_x1=-1
upper_y0=110
upper_y1=135
lower_y0=145
lower_y1=175
roi_width=96
roi_height=96
roi_gap_above_upper_zone=0
x_smoothing_alpha=0.70
miss_tolerance_frames=5
zone_min_pixels=24
zone_min_ratio=0.015
zone_min_blob_area=20
zone_max_blob_area=5000
center_x_max_diff=24
trigger_consecutive_frames=2
```
