# Function inventory

## CLI / config

### `src/main.cpp`
- `trim`
- `parse_bool`
- `parse_roi_csv`
- `load_config`
- `print_help`
- `run_probe_mode`
- `main`

## Camera / probe

### `src/camera.hpp`
- `run_command`
- `probe_camera`
- `print_probe`
- `fourcc_from_string`
- `clamp_camera_size`
- `open_capture`
- `grab_latest_frame`
- `downscale_for_preview`
- `bench_capture`

## AprilTag / warp / ROI geometry

### `src/calibrate.hpp`
- `finite_pt`
- `quad_area4`
- `families_from_mode`
- `family_name_from_dict`
- `detect_apriltag_best`
- `TagLocker::reset`
- `TagLocker::update`
- `clamp_roi`
- `clamp_rect_xywh`
- `roi_to_rect`
- `normalize_roi_mode`
- `is_dynamic_roi_mode`
- `dynamic_search_rect`
- `dynamic_image_roi_rect`
- `save_rois_yaml`
- `load_rois_yaml`
- `build_centered_warp_package_from_detection_px`
- `build_warp_package_from_detection`
- `apply_warp`
- `save_warp_package`
- `load_warp_package`
- `draw_detection_overlay`
- `draw_rois`
- `adjust_roi`

## App flow / operator surface

### `src/deploy.hpp`
- `print_calibrate_controls`
- `make_blank_preview`
- `build_deploy_config_warning`
- `save_crop_if_needed`
- `red_status_text`
- `draw_runtime_roi_overlay`
- `run_calibrate`
- `run_deploy`

## Runtime ROI extraction

### `src/roi_helper.cpp`
- internal: `apply_mask_fill`
- internal: `threshold_red_hsv`
- internal: `build_red_mask_vis`
- `extract_runtime_rois`

## Model runtime facade

### `src/model.hpp`
- `init_model_runtime`
- `release_model_runtime`
- `run_model_on_image_roi`

### `src/model.cpp` internal helpers
- `parse_preprocess`
- `apply_mask_fill`

## Classifier-common utilities

### `src/classifier_common.hpp`
- `read_labels`
- `ensure_bgr`
- `estimate_border_color`
- `resize_stretch`
- `center_crop_square_resize`
- `letterbox_square_resize`
- `preprocess_image`
- `softmax`
- `looks_like_probability_distribution`
- `probabilities_from_output`
- `select_topk`
- `preprocess_mode_to_string`
- `preprocess_mode_from_string`
- `parse_triplet_csv`
- `is_image_file`
- `collect_images`

## ONNX backend

### `src/onnx_classifier.hpp`
- `OnnxClassifier::load`
- `OnnxClassifier::classify`

## NCNN backend

### `src/ncnn_classifier.hpp`
- `NcnnClassifier::load`
- `NcnnClassifier::classify`
- internal: `flatten_to_vector`
- internal: `infer_blob_names_from_param`

## Stats / report

### `src/stats.hpp`
- `print_runtime_stats`
- `write_report_md`
