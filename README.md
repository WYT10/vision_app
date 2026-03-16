# vision_app

Build every time with:

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

Modes:
- `probe`: show supported camera modes
- `bench`: test one camera config and measure real FPS
- `live`: detect AprilTag, show camera + warp windows, lock, edit ROIs, save warp/ROIs
- `deploy`: load saved warp and ROIs, compute red threshold, optionally pass image ROI into model

Notes:
- `red_roi` and `image_roi` are ratio rectangles in warp space
- invalid warped pixels remain visible as white and are masked out in ROI operations
- ONNX inference uses OpenCV DNN
- NCNN argument path is reserved, but this drop does not compile NCNN yet
