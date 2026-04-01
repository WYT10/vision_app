# Architecture

## 1. System goal

Build a vision runtime that is easy to tune, easy to explain, and easy to present.

The old system mixed these concerns together:
- capture
- calibration state
- ROI geometry
- red detection
- model execution
- text overlays

That coupling made it hard to tell whether a problem came from:
- the warp geometry
- the trigger logic
- the crop geometry
- the UI hiding information
- stale config / stale binary

## 2. Clean v2 pipeline

Use this explicit pipeline:

`camera -> tag lock -> warp -> stacked red trigger -> ROI synth -> optional model -> render/log`

Each stage should have a separate input/output contract.

## 3. Module split

### app layer
- `main.cpp`
- `modes.hpp/.cpp`

Responsibilities:
- load config
- select mode
- print effective config
- own top-level state machine

### config layer
- `params.hpp`
- `config.hpp/.cpp`

Responsibilities:
- define all parameters in one place
- load defaults
- load INI
- apply CLI overrides
- dump effective config
- resolve relative paths against config file dir

### trigger layer
- `red_trigger.hpp/.cpp`

Responsibilities:
- define stacked upper/lower red zones
- threshold red in warped image
- compute per-zone red mass
- compute per-zone x centers
- reject unstable / inconsistent detections
- output a trigger result object

### ROI synthesis layer
- `roi_runtime.hpp/.cpp`

Responsibilities:
- fixed mode: convert saved rectangles into runtime rects
- dynamic mode: create image ROI above upper red zone from trigger result

### render / debug layer
- `text_console.hpp/.cpp`
- later `visualize.hpp/.cpp`

Responsibilities:
- keep geometry on image windows
- keep verbose text in a separate wrapped text window

## 4. Stacked-zone trigger geometry

Dynamic mode defines:
- one upper horizontal zone
- one lower horizontal zone
- shared x search range

Trigger only arms when:
- upper zone passes
- lower zone passes
- upper and lower x centers are consistent
- consecutive-frame persistence passes

Then:
- `x_center = 0.5 * (x_upper + x_lower)`
- image ROI is placed above the upper zone

## 5. Why stacked zones are better here

This matches your intended physical relation:
- red structure is lower in the warped view
- image ROI is above it
- trigger comes from a vertical relation, not just one patch of red

This reduces accidental triggering from isolated red noise because the system requires red structure to exist in two vertically separated places.
