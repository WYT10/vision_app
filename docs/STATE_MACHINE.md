# State machines

## Calibrate

### Search
- grab frame
- detect tag
- if tag stable, allow lock
- show temporary warp preview

### Locked
- apply warp
- evaluate selected ROI pipeline
- show trigger geometry
- allow live tuning keys
- save effective profile on demand

### Unlock
- clear lock
- return to Search

## Deploy

### Init
- load warp package
- load runtime profile
- open camera

### Loop
- grab frame
- apply warp
- run trigger analysis
- synthesize ROI
- if trigger stable and model enabled: run model
- update windows + text console

## Dynamic stacked trigger sub-state

### NotReady
Conditions not met.
- upper zone insufficient, or
- lower zone insufficient, or
- x mismatch, or
- persistence not met

Output:
- no final ROI
- `trigger_ready = false`

### Arming
Both zones pass this frame but persistence count is not enough yet.

Output:
- provisional x is visible for debugging
- `trigger_ready = false`

### Ready
Both zones pass for `trigger_consecutive_frames` frames and x centers are consistent.

Output:
- `trigger_ready = true`
- final ROI synthesized above upper zone

### Lost
Trigger was ready but one or more gates fail.

Behavior:
- optionally hold last x for `miss_tolerance_frames`
- then return to NotReady
