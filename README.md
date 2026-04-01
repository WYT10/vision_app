# vision_app completed detailed v2

This pack resets the app around a clearer machine:

`camera -> warp -> trigger -> roi synth -> optional model -> UI/report`

## What is completed here

- real CLI mode dispatch
- config file loading with CLI override precedence
- inline `# comment` stripping in config
- relative config paths resolved relative to the config file location
- `probe` tasks:
  - `list`
  - `live`
  - `snap`
  - `bench`
- two ROI modes:
  - `fixed`
  - `dynamic_red_stacked`
- dynamic trigger logic using **two stacked horizontal red zones**
- dynamic image ROI synthesized **above the upper zone**
- wrapped independent text window: `vision_app_text`
- optional red mask window: `vision_app_red_mask`
- warp center placement control via:
  - `warp_center_x_ratio`
  - `warp_center_y_ratio`

## What is still limited

- not compile-verified in this environment
- dynamic parameters are only saved through config/report, not a dedicated dynamic YAML profile yet
- model path remains optional and disabled by default

## Build

```bash
export CMAKE_PREFIX_PATH=/home/pi/ncnn/build/install:$CMAKE_PREFIX_PATH
cd ~/Desktop/vision_app
rm -rf build
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Typical runs

Probe formats:

```bash
./vision_app --mode probe --probe-task list --device /dev/video0
```

Probe benchmark:

```bash
./vision_app --mode probe --probe-task bench --device /dev/video0 --duration 5
```

Calibrate stacked dynamic mode:

```bash
./vision_app --config ../configs/vision_app.conf --mode calibrate --roi-mode dynamic_red_stacked --red-show-mask-window 1
```

Deploy stacked dynamic mode:

```bash
./vision_app --config ../configs/vision_app.conf --mode deploy --roi-mode dynamic_red_stacked
```


## Folder layout

```
vision_app/
├── CMakeLists.txt
├── configs/
│   └── vision_app.conf
├── docs/
├── include/vision_app/
│   ├── camera.hpp
│   ├── calibrate.hpp
│   ├── classifier_common.hpp
│   ├── deploy.hpp
│   ├── model.hpp
│   ├── ncnn_classifier.hpp
│   ├── onnx_classifier.hpp
│   ├── stats.hpp
│   └── text_console.hpp
├── src/
│   ├── main.cpp
│   ├── model.cpp
│   └── roi_helper.cpp
└── cmake/
```
