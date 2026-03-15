#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <string>
#include <vector>

/*
==============================================================================
config.h
==============================================================================
Purpose
    Shared configuration contract for the whole application.

Design notes
    - Keep the runtime state in small plain structs.
    - Keep the on-disk JSON shape close to these structs.
    - Keep camera mode identical between calibration and deploy.

Debugging notes
    - If calibration and deploy disagree, check CameraMode first.
    - If deploy refuses to start, compare requested_mode with
      calibration.camera_mode_used.
==============================================================================
*/

/*
------------------------------------------------------------------------------
ROI ratio
------------------------------------------------------------------------------
Input space
    Pixel rectangle inside a warped image.

Stored form
    Normalized x/y/w/h in the range [0, 1] relative to warped output size.

Why
    The warped image size can change between calibrations, so ratios are more
    stable than storing raw pixels.
------------------------------------------------------------------------------
*/
struct RoiRatio
{
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
};

/*
------------------------------------------------------------------------------
Camera mode
------------------------------------------------------------------------------
Fields
    width   : requested or actual frame width in pixels
    height  : requested or actual frame height in pixels
    fps     : requested or actual frame rate
    fourcc  : pixel format such as MJPG or YUYV
------------------------------------------------------------------------------
*/
struct CameraMode
{
    int width = 320;
    int height = 240;
    int fps = 60;
    std::string fourcc = "MJPG";
};

/*
------------------------------------------------------------------------------
Camera config
------------------------------------------------------------------------------
requested_mode
    The single active camera mode used by both calibration and deploy.

probe_candidates
    Candidate modes tested by the probe tool.

buffer_size
    Requested backend queue length. Smaller values reduce latency when the
    backend respects the hint.

warmup_frames
    Frames discarded after open so FPS measurement is less misleading.

flip_horizontal
    Optional debug convenience for mirrored camera mounts.
------------------------------------------------------------------------------
*/
struct CameraConfig
{
    int device_index = 0;
    std::string device_path;
    std::string v4l2_device;
    int backend = cv::CAP_V4L2;
    CameraMode requested_mode;
    int buffer_size = 1;
    int warmup_frames = 20;
    int drop_frames_per_read = 0;
    bool flip_horizontal = false;
    std::vector<CameraMode> probe_candidates {
        {640, 480, 60, "MJPG"},
        {320, 240, 60, "MJPG"},
        {160, 120, 120, "MJPG"},
        {160, 120, 180, "MJPG"}
    };
};

/*
------------------------------------------------------------------------------
Tag detection config
------------------------------------------------------------------------------
family_mode
    "auto" -> try several families
    otherwise use allowed_family only

allowed_id
    -1 means accept any ID in the selected family set.
------------------------------------------------------------------------------
*/
struct TagConfig
{
    std::string family_mode = "auto";
    std::string allowed_family;
    int allowed_id = -1;
    double tag_size_units = 100.0;
    double output_padding_units = 10.0;
    bool lock_on_first_detection = false;
};

/*
------------------------------------------------------------------------------
Trigger config
------------------------------------------------------------------------------
Trigger rule
    R > red_threshold
    R > G + red_margin
    R > B + red_margin

cooldown_ms
    Minimum spacing between saved trigger events.
------------------------------------------------------------------------------
*/
struct TriggerConfig
{
    double red_threshold = 180.0;
    double red_margin = 40.0;
    int cooldown_ms = 500;
    bool save_raw = true;
    bool save_warped = true;
    bool save_roi = true;
    std::string capture_dir = "captures";
};

/*
------------------------------------------------------------------------------
Runtime config
------------------------------------------------------------------------------
show_ui
    Master switch for windows and overlays.

headless_deploy
    Extra guard so deploy can stay minimal even when show_ui is true in config.

probe_measure_frames
    Number of frames used for real FPS measurement after warmup.
------------------------------------------------------------------------------
*/
struct RuntimeConfig
{
    bool show_ui = true;
    bool headless_deploy = false;
    int probe_measure_frames = 120;
    std::string report_dir = "reports";
};

/*
------------------------------------------------------------------------------
Calibration payload
------------------------------------------------------------------------------
valid
    True only after a successful save from calibration mode.

camera_mode_used
    Exact camera mode that produced the saved homography and ROI data.

homography
    3x3 perspective transform from raw frame to warped ground plane.
------------------------------------------------------------------------------
*/
struct CalibrationData
{
    bool valid = false;
    CameraMode camera_mode_used;
    cv::Mat homography;
    int warped_width = 0;
    int warped_height = 0;
    RoiRatio red_roi_ratio;
    RoiRatio image_roi_ratio;
};

struct AppConfig
{
    CameraConfig camera;
    TagConfig tag;
    TriggerConfig trigger;
    RuntimeConfig runtime;
    CalibrationData calibration;
};

/*
------------------------------------------------------------------------------
JSON IO helpers
------------------------------------------------------------------------------
load_config
    Input  : JSON file path
    Output : populated AppConfig
    Return : true on success

save_config
    Input  : JSON file path + AppConfig
    Output : file written to disk
    Return : true on success
------------------------------------------------------------------------------
*/
bool load_config(const std::string& path, AppConfig& config, std::string* error = nullptr);
bool save_config(const std::string& path, const AppConfig& config, std::string* error = nullptr);

/* Backend and mode utilities. */
std::string backend_to_string(int backend);
int backend_from_string(const std::string& backend_name);
std::string normalize_fourcc_string(const std::string& fourcc);
bool camera_modes_match(const CameraMode& a, const CameraMode& b);
