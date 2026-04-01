# Function map

## 1. Config / app

### `bool load_default_config(AppConfig&, std::string& err)`
Create a complete default config.

### `bool load_config_ini(const std::string& path, AppConfig&, std::string& err)`
Load INI-like config file.

### `bool apply_cli_overrides(int argc, char** argv, AppConfig&, std::string& err)`
Override config after file load.

### `std::string dump_effective_config(const AppConfig&)`
Return a human-readable summary of the actual loaded settings.

## 2. Trigger analysis

### `bool compute_red_mask_stacked(const cv::Mat& warped, const cv::Mat& valid_mask, const DynamicStackedConfig& dyn, const RedThresholdConfig& thr, StackedRedDebug& out, std::string& err)`
Create the red mask and measure upper/lower zone stats.

Outputs:
- upper zone rectangle
- lower zone rectangle
- binary red mask in full warped coordinates
- upper zone stats
- lower zone stats
- full-band stats
- upper/lower blob x centers

### `bool evaluate_stacked_trigger(const StackedRedDebug& dbg, const DynamicStackedConfig& dyn, const RedThresholdConfig& thr, TriggerState& state, TriggerResult& out, std::string& err)`
Turn one-frame measurements into a trigger decision.

Responsibilities:
- per-zone pass/fail
- x consistency gate
- consecutive-frame persistence
- miss tolerance / optional hold-last-x behavior

## 3. ROI synthesis

### `bool synthesize_fixed_rois(const cv::Size& warped_size, const FixedRoiConfig&, RuntimeRoiResult& out, std::string& err)`
Build old fixed rectangles.

### `bool synthesize_dynamic_roi_above_upper(const cv::Size& warped_size, const DynamicStackedConfig& dyn, const TriggerResult& trig, RuntimeRoiResult& out, std::string& err)`
Create the image ROI above the upper zone.

Geometry:
- `roi_bottom = upper_y0 - roi_gap_above_upper_zone`
- `roi_top = roi_bottom - roi_height`
- `roi_left = x_center - roi_width/2`
- `roi_right = roi_left + roi_width`

## 4. Text window

### `TextConsole::show(const ConsoleSnapshot&)`
Render a wrapped text console in a separate OpenCV window.

Content should include:
- session summary
- current mode / roi mode
- trigger metrics
- geometry values
- warnings
- controls

## 5. Later runtime functions

### `bool run_probe(const AppConfig&, std::string& err)`
Tasks:
- list camera modes
- live preview
- snap image
- FPS bench

### `bool run_calibrate(const AppConfig&, std::string& err)`
Tasks:
- tag search/lock
- warp preview
- dynamic geometry tuning
- save effective runtime profile

### `bool run_deploy(const AppConfig&, std::string& err)`
Tasks:
- load warp/profile
- warp
- evaluate trigger
- synthesize ROI
- optionally run model after stable trigger
