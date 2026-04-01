# Visual and interaction design

## Window A — `vision_app_camera`

Purpose:
- verify camera exposure / focus / framing
- verify AprilTag visibility and lock state

Show:
- raw camera frame
- tag corners and id/family
- lock status
- camera FPS / mode summary

## Window B — `vision_app_warp`

This is the main operator window.

### Fixed mode overlays
- fixed red ROI rectangle
- fixed image ROI rectangle

### Dynamic stacked mode overlays
- upper red zone rectangle
- lower red zone rectangle
- accepted red blob box / center in upper zone
- accepted red blob box / center in lower zone
- `x_upper`
- `x_lower`
- final `x_center`
- vertical line at final `x_center`
- synthesized image ROI above upper zone
- valid/invalid warp area indication

## Window C — `vision_app_red_mask`

Purpose:
- truth window for threshold tuning

Show:
- binary red mask in warped coordinates
- upper/lower zone rectangles
- accepted blobs highlighted
- rejected blobs can be drawn lightly if useful

## Window D — `vision_app_text`

Purpose:
- remove text clutter from image overlays
- present metrics like a terminal / status dashboard

Suggested content sections:

### Session
- mode
- roi mode
- config path
- save/load paths

### Geometry
- warp size
- upper zone `[x0:x1, y0:y1]`
- lower zone `[x0:x1, y0:y1]`
- ROI size and gap

### Trigger
- upper red pixels / ratio / x
- lower red pixels / ratio / x
- band red pixels / ratio
- `x_center`
- `trigger_ready`
- persistence count

### Controls
- live tuning keys

### Warnings
- ROI clipped by edge
- invalid warp margin too small
- trigger lost
- camera / warp size mismatch

## Live tuning key layout (recommended)

### Zone movement
- `w / s` move both zones up/down
- `t / g` upper zone height `+ / -`
- `y / h` lower zone height `+ / -`
- `i / k` gap between upper and lower zones `+ / -`

### ROI geometry
- `a / d` ROI width `- / +`
- `z / x` ROI height `- / +`
- `j / l` gap above upper zone `- / +`

### Trigger strictness
- `c / v` zone min pixels `- / +`
- `f / b` zone min ratio `- / +`
- `n / m` x consistency tolerance `- / +`

### General
- `[` / `]` step size down/up
- `tab` switch tuning group
- `p` save profile
- `r` reset current dynamic params
- `q` quit

## Interaction principle

Do not use image windows as text dashboards.
- image windows should communicate geometry
- text window should communicate values and logs

That separation makes the system easier to debug and easier to present.
