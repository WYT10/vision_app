# Optimization notes

## Highest-value practical optimizations

### 1. Keep the pipeline separable
Do not merge dynamic red-center logic into the current fixed ROI path.
Maintain two explicit modes:

- fixed saved ROI runtime
- dynamic red-center runtime

That preserves baseline comparability.

### 2. Constrain computation early
For the future dynamic red path:
- detect red only in a vertical band `[a:b]`
- optionally restrict x search too
- compute only the blob center-x
- build final crop from `(x_center, a, roi_w, roi_h)`

### 3. Keep camera latency low
Recommended defaults on Pi:
- `latest_only=1`
- `buffer_size=1`
- `drain_grabs=1`
- low preview sizes

### 4. Avoid benchmark contamination
When measuring model speed:
- use `--headless`
- keep save-to-disk off
- benchmark one backend at a time
- keep preprocess identical when comparing ONNX vs NCNN

### 5. Add structured benchmarks next
Useful deploy report fields:
- camera mode
- warp ms avg/min/max
- ROI ms avg/min/max
- model ms avg/min/max
- total FPS avg/min/max
- dropped / skipped model frames
- backend + preprocess + thread count
