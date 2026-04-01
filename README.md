# vision_app v2 — stacked red-zone blueprint

This pack resets the design around the geometry you described:

- one **upper horizontal red zone**
- one **lower horizontal red zone**
- both zones must pass before the trigger is armed
- `x_center` is computed from the two zones
- the dynamic image ROI is synthesized **above the upper zone**

This pack is meant to make the system understandable and implementable.
It is a **clean architecture + starter implementation**, not a full drop-in finished app.

## Core runtime model

`camera -> tag lock -> warp -> stacked red trigger -> dynamic ROI synth -> optional model -> UI/log`

## Two ROI modes

### 1. `fixed`
Old baseline mode.
- saved `red_roi`
- saved `image_roi`
- deterministic rectangles
- useful for baseline / comparison

### 2. `dynamic_red_stacked`
New adaptive mode.
- upper red zone
- lower red zone
- both zones must pass red thresholds
- x-center computed from both zones
- final ROI placed above the upper zone

## Windows

- `vision_app_camera` — raw feed + tag / camera state
- `vision_app_warp` — main warped view with trigger geometry and final ROI
- `vision_app_red_mask` — binary red mask debug
- `vision_app_text` — wrapped text console instead of overlay spam

## What is implemented in this pack

- explicit parameter model for stacked trigger zones
- starter C++ implementation for:
  - red mask extraction
  - per-zone measurement
  - trigger evaluation
  - dynamic ROI synthesis above the upper zone
- wrapped text console helper
- docs for architecture, state machine, parameter reference, and UI interaction

## What is intentionally not finished here

- full AprilTag detection / warp lock loop
- full camera probing loop
- full save/load config parser
- full deploy loop

Those are kept out so the trigger geometry is easy to reason about first.
