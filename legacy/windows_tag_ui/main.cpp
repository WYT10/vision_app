#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

// ─── Constants ───────────────────────────────────────────────────────────────
static constexpr const char *kWindowName = "AprilTag Camera";
static constexpr const char *kDefaultCalibrationStem = "camera_plot_tag_calibration";

// Fixed panel container size – camera image is scaled to fill panel 1
static constexpr int kPanelW = 320;
static constexpr int kPanelH = 240;
static constexpr int kTargetW = 160;
static constexpr int kTargetH = 120;
static constexpr int kStatusH = 30;
static constexpr int kPickerMaxVisible = 8;

// Button rects inside panel 1 (4 buttons spread across kPanelW, placed near bottom)
static const cv::Rect kLockRect(9, 200, 68, 32);
static const cv::Rect kClearRect(86, 200, 68, 32);
static const cv::Rect kSaveRect(163, 200, 68, 32);
static const cv::Rect kLoadRect(240, 200, 68, 32);

static const std::map<std::string, int> kDictionaries = {
    {"AprilTag 16h5", cv::aruco::DICT_APRILTAG_16h5},
    {"AprilTag 25h9", cv::aruco::DICT_APRILTAG_25h9},
    {"AprilTag 36h10", cv::aruco::DICT_APRILTAG_36h10},
    {"AprilTag 36h11", cv::aruco::DICT_APRILTAG_36h11},
};

// ─── Structures ──────────────────────────────────────────────────────────────
struct RectificationCalibration
{
    std::string family_name;
    int marker_id = -1;
    std::vector<cv::Point2f> display_corners;
    cv::Mat full_frame_transform; // 3×3 CV_64F
    int output_width = 0;
    int output_height = 0;
    int frame_width = 0;
    int frame_height = 0;
    bool flip_horizontal = false;
};

struct DetectorEntry
{
    std::string name;
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    cv::Ptr<cv::aruco::DetectorParameters> params;
};

struct Detection
{
    std::string family_name;
    int marker_id = -1;
    std::vector<cv::Point2f> corners;
};

struct DisplayInput
{
    int key = -1;
    std::optional<cv::Point> click;
};

struct UiState
{
    bool load_picker_open = false;
    std::vector<fs::path> calibration_files;
    int selected_calibration = 0;
    int picker_scroll = 0;
    std::string status_text = "Ready";
    double fps_ema = 0.0;
};

RectificationCalibration loadCalibration(const fs::path &path);
std::vector<fs::path> listCalibrationFiles(const fs::path &dir);

struct Options
{
    int camera = 0;
    int search_limit = 5;
    std::string family = "auto";
    bool flip_horizontal = true;
    bool freeze_calibration = true;
    bool auto_lock = false;
    std::string save_calibration;
    std::string load_calibration;
    std::string calibration_file;
    bool auto_load_calibration = true;
    bool auto_save_calibration = true;
};

// ─── Mouse callback ───────────────────────────────────────────────────────────
static std::optional<cv::Point> g_pending_click;
static void onMouse(int event, int x, int y, int, void *)
{
    if (event == cv::EVENT_LBUTTONDOWN)
        g_pending_click = cv::Point(x, y);
}

// ─── Argument parsing ─────────────────────────────────────────────────────────
Options parseArgs(int argc, char **argv)
{
    Options o;
    for (int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto req = [&]() -> std::string
        {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + a);
            return argv[++i];
        };
        if (a == "--camera")
            o.camera = std::stoi(req());
        else if (a == "--search-limit")
            o.search_limit = std::stoi(req());
        else if (a == "--family")
            o.family = req();
        else if (a == "--flip-horizontal")
            o.flip_horizontal = true;
        else if (a == "--no-flip-horizontal")
            o.flip_horizontal = false;
        else if (a == "--no-freeze-calibration")
            o.freeze_calibration = false;
        else if (a == "--auto-lock")
            o.auto_lock = true;
        else if (a == "--save-calibration")
            o.save_calibration = req();
        else if (a == "--load-calibration")
            o.load_calibration = req();
        else if (a == "--calibration-file")
            o.calibration_file = req();
        else if (a == "--no-auto-load-calibration")
            o.auto_load_calibration = false;
        else if (a == "--no-auto-save-calibration")
            o.auto_save_calibration = false;
        else if (a == "--help" || a == "-h")
        {
            std::cout << "Usage: camera_plot_tag_cpp [options]\n"
                         "  --camera N                    Camera index (default: 0)\n"
                         "  --search-limit N              Indices to probe (default: 5)\n"
                         "  --family F                    AprilTag family or 'auto' (default: auto)\n"
                         "  --flip-horizontal             Mirror image (default: on)\n"
                         "  --no-flip-horizontal          Disable mirror\n"
                         "  --no-freeze-calibration       Recalculate every frame\n"
                         "  --auto-lock                   Lock on first detection\n"
                         "  --save-calibration PATH       Save calibration JSON\n"
                         "  --load-calibration PATH       Load calibration JSON\n"
                         "  --calibration-file PATH       Base path for calibration files\n"
                         "  --no-auto-load-calibration    Disable auto-load\n"
                         "  --no-auto-save-calibration    Disable auto-save\n";
            std::exit(0);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + a);
        }
    }
    return o;
}

// ─── Camera ───────────────────────────────────────────────────────────────────
cv::VideoCapture openCamera(int preferred, int limit)
{
    std::vector<int> indices;
    if (preferred >= 0)
    {
        indices.push_back(preferred);
        for (int i = 0; i < limit; ++i)
            if (i != preferred)
                indices.push_back(i);
    }
    else
    {
        for (int i = 0; i < limit; ++i)
            indices.push_back(i);
    }

    for (int idx : indices)
    {
        for (int backend : {(int)cv::CAP_V4L2, (int)cv::CAP_ANY})
        {
            cv::VideoCapture cap(idx, backend);
            if (!cap.isOpened())
                continue;
            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
            // Request smallest common resolution; we will center-crop to 160×120 after capture
            cap.set(cv::CAP_PROP_FRAME_WIDTH, kTargetW);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, kTargetH);
            cap.set(cv::CAP_PROP_FPS, 30);
            int actual_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
            int actual_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
            std::cout << "Opened camera index " << idx
                      << " (" << actual_w << "x" << actual_h << ")\n";
            return cap;
        }
    }
    throw std::runtime_error("Could not open a camera.");
}

// ─── Detection ────────────────────────────────────────────────────────────────
std::vector<cv::Point2f> scaleCorners(const std::vector<cv::Point2f> &c, float sx, float sy)
{
    std::vector<cv::Point2f> out;
    out.reserve(c.size());
    for (const auto &p : c)
        out.emplace_back(p.x * sx, p.y * sy);
    return out;
}

std::vector<Detection> detectTags(const cv::Mat &gray,
                                  const std::vector<DetectorEntry> &detectors)
{
    std::vector<Detection> out;
    out.reserve(detectors.size());
    for (const auto &e : detectors)
    {
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> rejected;
        cv::aruco::detectMarkers(gray, e.dictionary, corners, ids, e.params, rejected);
        for (size_t i = 0; i < ids.size(); ++i)
            out.push_back({e.name, ids[i], corners[i]});
    }
    return out;
}

// ─── Overlay ──────────────────────────────────────────────────────────────────
void drawDetectionOverlay(cv::Mat &frame, const std::string &family, int id,
                          const std::vector<cv::Point2f> &corners)
{
    if (corners.size() != 4)
        return;
    std::vector<cv::Point> pts;
    for (const auto &c : corners)
        pts.emplace_back(cvRound(c.x), cvRound(c.y));
    const cv::Point *p = pts.data();
    int n = 4;
    cv::polylines(frame, &p, &n, 1, true, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    const char *lbl[] = {"0", "1", "2", "3"};
    for (int i = 0; i < 4; ++i)
    {
        cv::circle(frame, pts[i], 6, cv::Scalar(0, 0, 255), -1);
        cv::putText(frame, lbl[i], cv::Point(pts[i].x + 8, pts[i].y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }
    cv::putText(frame, family + " | id=" + std::to_string(id),
                cv::Point(cvRound(corners[0].x), cvRound(corners[0].y) - 12),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
}

// ─── Buttons ──────────────────────────────────────────────────────────────────
void drawButton(cv::Mat &f, const cv::Rect &r, const std::string &label, bool enabled, bool active)
{
    cv::Scalar fill = active ? cv::Scalar(255, 140, 0) : (enabled ? cv::Scalar(235, 235, 235) : cv::Scalar(205, 205, 205));
    cv::Scalar text = active ? cv::Scalar(255, 255, 255) : (enabled ? cv::Scalar(35, 35, 35) : cv::Scalar(120, 120, 120));
    cv::Scalar border = active ? cv::Scalar(220, 110, 0) : cv::Scalar(160, 160, 160);
    cv::rectangle(f, r, fill, -1);
    cv::rectangle(f, r, border, 2);
    int bl;
    cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &bl);
    cv::putText(f, label,
                cv::Point(r.x + (r.width - ts.width) / 2, r.y + (r.height + ts.height) / 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, text, 2, cv::LINE_AA);
}

void drawControls(cv::Mat &f, bool locked, bool can_lock)
{
    drawButton(f, kLockRect, "Lock [L]", can_lock, locked);
    drawButton(f, kClearRect, "Clear [C]", locked, false);
    drawButton(f, kSaveRect, "Save [S]", locked, false);
    drawButton(f, kLoadRect, "Load [O]", true, false);
}

bool inRect(const cv::Point &pt, const cv::Rect &r) { return r.contains(pt); }

bool shouldLock(const DisplayInput &di, bool can_lock)
{
    if (!can_lock)
        return false;
    if (di.key == 'l')
        return true;
    return di.click && inRect(*di.click, kLockRect);
}
bool shouldClear(const DisplayInput &di, bool locked)
{
    if (!locked)
        return false;
    if (di.key == 'c')
        return true;
    return di.click && inRect(*di.click, kClearRect);
}
bool shouldSave(const DisplayInput &di, bool locked)
{
    if (!locked)
        return false;
    if (di.key == 's')
        return true;
    return di.click && inRect(*di.click, kSaveRect);
}
bool shouldLoad(const DisplayInput &di)
{
    if (di.key == 'o')
        return true;
    return di.click && inRect(*di.click, kLoadRect);
}

bool shouldConfirm(const DisplayInput &di)
{
    return di.key == 13 || di.key == 10;
}

bool shouldCancel(const DisplayInput &di)
{
    return di.key == 27;
}

// ─── Corner plot panel ────────────────────────────────────────────────────────
cv::Mat buildCornerPlot(const std::vector<cv::Point2f> *corners, int w, int h)
{
    cv::Mat plot(h, w, CV_8UC3, cv::Scalar(245, 245, 245));
    cv::rectangle(plot, {20, 20}, {w - 20, h - 20}, cv::Scalar(210, 210, 210), 1);
    cv::putText(plot, "Corner Plot", {20, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(30, 30, 30), 2, cv::LINE_AA);

    if (!corners || corners->empty())
    {
        cv::putText(plot, "Show a printed AprilTag to the camera",
                    {20, h / 2 - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
        cv::putText(plot, "Points are ordered 0, 1, 2, 3",
                    {20, h / 2 + 20}, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
        return plot;
    }

    float mnx = (*corners)[0].x, mny = (*corners)[0].y, mxx = mnx, mxy = mny;
    for (const auto &p : *corners)
    {
        mnx = std::min(mnx, p.x);
        mny = std::min(mny, p.y);
        mxx = std::max(mxx, p.x);
        mxy = std::max(mxy, p.y);
    }
    float spx = std::max(mxx - mnx, 1.f), spy = std::max(mxy - mny, 1.f);
    int left = 50, top = 60, dw = w - 100, dh = h - 120;
    std::vector<cv::Point> pts;
    for (const auto &p : *corners)
        pts.emplace_back(left + (int)((p.x - mnx) / spx * dw), top + (int)((p.y - mny) / spy * dh));

    const cv::Point *pp = pts.data();
    int n = 4;
    cv::polylines(plot, &pp, &n, 1, true, cv::Scalar(255, 140, 0), 2);
    const char *lbl[] = {"0", "1", "2", "3"};
    for (int i = 0; i < 4; ++i)
    {
        cv::circle(plot, pts[i], 7, cv::Scalar(220, 40, 40), -1);
        cv::putText(plot, lbl[i], {pts[i].x + 10, pts[i].y - 10},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(20, 20, 20), 2, cv::LINE_AA);
    }
    const char *pl[] = {"P0", "P1", "P2", "P3"};
    for (int i = 0; i < 4; ++i)
    {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s: (%.1f, %.1f)", pl[i], (*corners)[i].x, (*corners)[i].y);
        cv::putText(plot, buf, {20, h - 20 - (3 - i) * 22},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    }
    return plot;
}

// ─── Rectified view panel ─────────────────────────────────────────────────────
cv::Mat buildRectifiedView(const cv::Mat &frame, const RectificationCalibration *cal,
                           int w, int h, cv::Mat *warped_out = nullptr)
{
    cv::Mat panel(h, w, CV_8UC3, cv::Scalar(245, 245, 245));
    cv::rectangle(panel, {20, 20}, {w - 20, h - 20}, cv::Scalar(210, 210, 210), 1);
    cv::putText(panel, "Rectified Frame", {20, 35}, cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(30, 30, 30), 2, cv::LINE_AA);

    if (!cal)
    {
        cv::putText(panel, "Calibration pending", {20, h / 2 - 10},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
        cv::putText(panel, "Show an AprilTag to lock homography", {20, h / 2 + 20},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
        return panel;
    }
    if (frame.cols != cal->frame_width || frame.rows != cal->frame_height)
    {
        cv::putText(panel, "Calibration size mismatch", {20, h / 2 - 10},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        return panel;
    }

    cv::Mat warped;
    cv::warpPerspective(frame, warped, cal->full_frame_transform,
                        {cal->output_width, cal->output_height});
    if (warped_out)
        *warped_out = warped;

    int mx = 20, mt = 55, mb = 65;
    int aw = w - 2 * mx, ah = h - mt - mb;
    double sc = std::min((double)aw / cal->output_width, (double)ah / cal->output_height);
    int sw = std::max((int)std::round(cal->output_width * sc), 1);
    int sh = std::max((int)std::round(cal->output_height * sc), 1);
    cv::Mat resized;
    cv::resize(warped, resized, {sw, sh}, 0, 0, cv::INTER_LINEAR);
    int xo = (w - sw) / 2, yo = mt + (ah - sh) / 2;
    resized.copyTo(panel(cv::Rect(xo, yo, sw, sh)));
    cv::rectangle(panel, {xo - 1, yo - 1}, {xo + sw, yo + sh}, cv::Scalar(255, 140, 0), 2);
    char buf[64];
    std::snprintf(buf, sizeof(buf), "Warp size: %d x %d", cal->output_width, cal->output_height);
    cv::putText(panel, buf, {20, h - 22}, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    return panel;
}

// ─── UI helpers ───────────────────────────────────────────────────────────────
int normalizeKey(int key)
{
    if (key < 0)
        return key;
    key &= 0xFF;
    return std::tolower(key);
}

cv::Mat prepareCaptureFrame(const cv::Mat &frame)
{
    if (frame.empty())
        return frame;

    const double target_aspect = static_cast<double>(kTargetW) / kTargetH;
    const double frame_aspect = static_cast<double>(frame.cols) / std::max(frame.rows, 1);

    int crop_x = 0;
    int crop_y = 0;
    int crop_w = frame.cols;
    int crop_h = frame.rows;

    if (frame_aspect > target_aspect)
    {
        crop_w = std::max(1, static_cast<int>(std::round(frame.rows * target_aspect)));
        crop_x = std::max((frame.cols - crop_w) / 2, 0);
    }
    else if (frame_aspect < target_aspect)
    {
        crop_h = std::max(1, static_cast<int>(std::round(frame.cols / target_aspect)));
        crop_y = std::max((frame.rows - crop_h) / 2, 0);
    }

    cv::Mat cropped = frame(cv::Rect(crop_x, crop_y, crop_w, crop_h));
    if (cropped.cols == kTargetW && cropped.rows == kTargetH)
        return cropped.clone();

    cv::Mat resized;
    cv::resize(cropped, resized, {kTargetW, kTargetH}, 0, 0, cv::INTER_AREA);
    return resized;
}

fs::path determineCalibrationDir(const Options &args)
{
    auto pick_parent = [](const std::string &raw) -> fs::path
    {
        if (raw.empty())
            return {};
        fs::path path(raw);
        if (path.has_parent_path())
            return path.parent_path();
        return fs::current_path();
    };

    if (auto dir = pick_parent(args.load_calibration); !dir.empty())
        return dir;
    if (auto dir = pick_parent(args.save_calibration); !dir.empty())
        return dir;
    if (auto dir = pick_parent(args.calibration_file); !dir.empty())
        return dir;
    return fs::current_path();
}

void refreshCalibrationPicker(UiState &ui, const fs::path &dir)
{
    ui.calibration_files = listCalibrationFiles(dir);
    if (ui.calibration_files.empty())
    {
        ui.selected_calibration = 0;
        ui.picker_scroll = 0;
        return;
    }
    ui.selected_calibration = std::clamp(ui.selected_calibration, 0,
                                         static_cast<int>(ui.calibration_files.size()) - 1);
    ui.picker_scroll = std::clamp(ui.picker_scroll, 0,
                                  std::max(0, static_cast<int>(ui.calibration_files.size()) - kPickerMaxVisible));
}

std::string calibrationLabel(const fs::path &path)
{
    return path.filename().string();
}

struct LoadPickerLayout
{
    cv::Rect panel_rect;
    std::vector<cv::Rect> item_rects;
    cv::Rect load_rect;
    cv::Rect cancel_rect;
};

LoadPickerLayout buildLoadPickerLayout(int canvas_w, int canvas_h, size_t item_count)
{
    const int panel_w = std::min(620, canvas_w - 40);
    const int visible_items = std::min(static_cast<int>(item_count), kPickerMaxVisible);
    const int panel_h = std::min(canvas_h - 30, 150 + visible_items * 32);
    const int panel_x = (canvas_w - panel_w) / 2;
    const int panel_y = std::max((canvas_h - panel_h) / 2, 15);

    LoadPickerLayout layout;
    layout.panel_rect = cv::Rect(panel_x, panel_y, panel_w, panel_h);

    const int list_x = panel_x + 24;
    const int list_y = panel_y + 58;
    const int list_w = panel_w - 48;
    for (int i = 0; i < visible_items; ++i)
        layout.item_rects.emplace_back(list_x, list_y + i * 32, list_w, 26);

    layout.load_rect = cv::Rect(panel_x + panel_w - 198, panel_y + panel_h - 48, 82, 30);
    layout.cancel_rect = cv::Rect(panel_x + panel_w - 102, panel_y + panel_h - 48, 82, 30);
    return layout;
}

void drawLoadPicker(cv::Mat &canvas, const UiState &ui, const fs::path &latest_cal_file)
{
    cv::Mat shade(canvas.size(), canvas.type(), cv::Scalar(20, 20, 20));
    cv::addWeighted(shade, 0.55, canvas, 0.45, 0.0, canvas);

    LoadPickerLayout layout = buildLoadPickerLayout(canvas.cols, canvas.rows, ui.calibration_files.size());
    cv::rectangle(canvas, layout.panel_rect, cv::Scalar(248, 248, 248), -1);
    cv::rectangle(canvas, layout.panel_rect, cv::Scalar(70, 70, 70), 2);
    cv::putText(canvas, "Load Calibration", {layout.panel_rect.x + 22, layout.panel_rect.y + 30},
                cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(25, 25, 25), 2, cv::LINE_AA);
    cv::putText(canvas, "Pick a saved version. Use W/S, 1-9, Enter, or mouse.",
                {layout.panel_rect.x + 22, layout.panel_rect.y + 52},
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);

    if (ui.calibration_files.empty())
    {
        cv::putText(canvas, "No calibration files found in the active directory.",
                    {layout.panel_rect.x + 22, layout.panel_rect.y + 96},
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(60, 60, 60), 1, cv::LINE_AA);
    }

    for (size_t i = 0; i < layout.item_rects.size(); ++i)
    {
        const int file_index = ui.picker_scroll + static_cast<int>(i);
        const bool selected = file_index == ui.selected_calibration;
        const bool is_active = !latest_cal_file.empty() && ui.calibration_files[file_index] == latest_cal_file;
        cv::Scalar fill = selected ? cv::Scalar(255, 224, 184) : cv::Scalar(235, 235, 235);
        cv::Scalar border = is_active ? cv::Scalar(0, 148, 255) : cv::Scalar(190, 190, 190);
        cv::rectangle(canvas, layout.item_rects[i], fill, -1);
        cv::rectangle(canvas, layout.item_rects[i], border, is_active ? 2 : 1);
        std::string prefix = std::to_string(file_index + 1) + ". ";
        cv::putText(canvas, prefix + calibrationLabel(ui.calibration_files[file_index]),
                    {layout.item_rects[i].x + 10, layout.item_rects[i].y + 18},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(30, 30, 30), 1, cv::LINE_AA);
    }

    if ((int)ui.calibration_files.size() > kPickerMaxVisible)
    {
        std::string footer = cv::format("Showing %d-%d of %d",
                                        ui.picker_scroll + 1,
                                        std::min(ui.picker_scroll + kPickerMaxVisible,
                                                 (int)ui.calibration_files.size()),
                                        (int)ui.calibration_files.size());
        cv::putText(canvas, footer, {layout.panel_rect.x + 24, layout.panel_rect.y + layout.panel_rect.height - 28},
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(90, 90, 90), 1, cv::LINE_AA);
    }

    drawButton(canvas, layout.load_rect, "Load", !ui.calibration_files.empty(), false);
    drawButton(canvas, layout.cancel_rect, "Cancel", true, false);
}

cv::Mat buildStatusBar(int width, const UiState &ui, const fs::path &latest_cal_file,
                       bool locked, bool picker_open)
{
    cv::Mat bar(kStatusH, width, CV_8UC3, cv::Scalar(38, 38, 38));
    std::string left = ui.status_text;
    if (picker_open)
        left = "Picker open: choose a calibration version";
    cv::putText(bar, left, {14, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(245, 245, 245), 1, cv::LINE_AA);

    std::string right = cv::format("FPS %.1f | %s", ui.fps_ema,
                                   locked ? calibrationLabel(latest_cal_file).c_str() : "unlocked");
    int baseline = 0;
    cv::Size text_size = cv::getTextSize(right, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
    cv::putText(bar, right, {std::max(14, width - text_size.width - 14), 20},
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
    return bar;
}

bool tryLoadCalibrationFile(const fs::path &path, const Options &args,
                            std::optional<RectificationCalibration> &calibration,
                            fs::path &latest_cal_file, UiState &ui)
{
    try
    {
        auto loaded = loadCalibration(path);
        if (loaded.flip_horizontal != args.flip_horizontal)
        {
            ui.status_text = "Load rejected: flip-horizontal mismatch";
            return false;
        }
        calibration = std::move(loaded);
        latest_cal_file = path;
        ui.status_text = "Loaded " + calibrationLabel(path);
        std::cout << "Loaded calibration from " << latest_cal_file << "\n";
        return true;
    }
    catch (const std::exception &e)
    {
        ui.status_text = std::string("Load failed: ") + e.what();
        std::cout << ui.status_text << "\n";
        return false;
    }
}

// ─── Calibration build ────────────────────────────────────────────────────────
RectificationCalibration buildCalibration(const cv::Mat &frame,
                                          const std::string &family,
                                          int marker_id,
                                          const std::vector<cv::Point2f> &corners,
                                          bool flip_h)
{
    float tw = cv::norm(corners[1] - corners[0]), bw = cv::norm(corners[2] - corners[3]);
    float lh = cv::norm(corners[3] - corners[0]), rh = cv::norm(corners[2] - corners[1]);
    int side = std::max((int)std::lround((tw + bw + lh + rh) / 4.f), 1);

    std::vector<cv::Point2f> dst = {
        {0.f, 0.f}, {(float)(side - 1), 0.f}, {(float)(side - 1), (float)(side - 1)}, {0.f, (float)(side - 1)}};
    cv::Mat T = cv::getPerspectiveTransform(corners, dst);

    int fw = frame.cols, fh = frame.rows;
    std::vector<cv::Point2f> outline = {
        {0.f, 0.f}, {(float)(fw - 1), 0.f}, {(float)(fw - 1), (float)(fh - 1)}, {0.f, (float)(fh - 1)}};
    std::vector<cv::Point2f> proj;
    cv::perspectiveTransform(outline, proj, T);

    float mnx = proj[0].x, mny = proj[0].y, mxx = mnx, mxy = mny;
    for (const auto &p : proj)
    {
        mnx = std::min(mnx, p.x);
        mny = std::min(mny, p.y);
        mxx = std::max(mxx, p.x);
        mxy = std::max(mxy, p.y);
    }
    double tx = -std::min(0.f, mnx), ty = -std::min(0.f, mny);
    cv::Mat trans = (cv::Mat_<double>(3, 3) << 1, 0, tx, 0, 1, ty, 0, 0, 1);

    RectificationCalibration cal;
    cal.family_name = family;
    cal.marker_id = marker_id;
    cal.display_corners = corners;
    cal.full_frame_transform = trans * T;
    cal.output_width = std::max((int)std::ceil(mxx + tx), side);
    cal.output_height = std::max((int)std::ceil(mxy + ty), side);
    cal.frame_width = fw;
    cal.frame_height = fh;
    cal.flip_horizontal = flip_h;
    return cal;
}

// ─── JSON save / load ─────────────────────────────────────────────────────────
void saveCalibration(const RectificationCalibration &cal, const fs::path &path)
{
    json cj = json::array();
    for (const auto &p : cal.display_corners)
        cj.push_back({p.x, p.y});
    json mj = json::array();
    for (int r = 0; r < 3; ++r)
    {
        json row = json::array();
        for (int c = 0; c < 3; ++c)
            row.push_back(cal.full_frame_transform.at<double>(r, c));
        mj.push_back(row);
    }
    json payload = {{"family_name", cal.family_name}, {"marker_id", cal.marker_id}, {"display_corners", cj}, {"full_frame_transform", mj}, {"output_width", cal.output_width}, {"output_height", cal.output_height}, {"frame_width", cal.frame_width}, {"frame_height", cal.frame_height}, {"flip_horizontal", cal.flip_horizontal}};
    std::ofstream f(path);
    if (!f)
        throw std::runtime_error("Cannot write: " + path.string());
    f << payload.dump(2) << '\n';
}

RectificationCalibration loadCalibration(const fs::path &path)
{
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("Cannot read: " + path.string());
    json p;
    f >> p;
    RectificationCalibration cal;
    cal.family_name = p.at("family_name").get<std::string>();
    cal.marker_id = p.at("marker_id").get<int>();
    cal.output_width = p.at("output_width").get<int>();
    cal.output_height = p.at("output_height").get<int>();
    cal.frame_width = p.at("frame_width").get<int>();
    cal.frame_height = p.at("frame_height").get<int>();
    cal.flip_horizontal = p.at("flip_horizontal").get<bool>();
    for (const auto &pt : p.at("display_corners"))
        cal.display_corners.emplace_back(pt.at(0).get<float>(), pt.at(1).get<float>());
    cal.full_frame_transform = cv::Mat::zeros(3, 3, CV_64F);
    int r = 0;
    for (const auto &row : p.at("full_frame_transform"))
    {
        int c = 0;
        for (const auto &v : row)
            cal.full_frame_transform.at<double>(r, c++) = v.get<double>();
        ++r;
    }
    return cal;
}

// ─── Calibration file helpers ─────────────────────────────────────────────────
std::vector<fs::path> listCalibrationFiles(const fs::path &dir)
{
    std::vector<fs::path> files;
    if (!fs::exists(dir))
        return files;
    for (const auto &e : fs::directory_iterator(dir))
    {
        const auto &p = e.path();
        if (p.extension() == ".json" && p.stem().string().rfind(kDefaultCalibrationStem, 0) == 0)
            files.push_back(p);
    }
    std::sort(files.begin(), files.end(), [](const fs::path &a, const fs::path &b)
              { return fs::last_write_time(a) > fs::last_write_time(b); });
    return files;
}

fs::path buildUniquePath(const std::string &save_arg, const std::string &file_arg)
{
    fs::path dir;
    std::string stem;
    if (!save_arg.empty())
    {
        fs::path p(save_arg);
        dir = p.parent_path().empty() ? fs::current_path() : p.parent_path();
        stem = p.stem().empty() ? kDefaultCalibrationStem : p.stem().string();
    }
    else if (!file_arg.empty())
    {
        fs::path p(file_arg);
        dir = p.parent_path().empty() ? fs::current_path() : p.parent_path();
        stem = p.stem().empty() ? kDefaultCalibrationStem : p.stem().string();
    }
    else
    {
        dir = fs::current_path();
        stem = kDefaultCalibrationStem;
    }
    fs::create_directories(dir);
    std::time_t t = std::time(nullptr);
    char ts[32];
    std::strftime(ts, sizeof(ts), "%Y%m%d_%H%M%S", std::localtime(&t));
    fs::path cand = dir / (stem + "_" + ts + ".json");
    for (int n = 1; fs::exists(cand); ++n)
        cand = dir / (stem + "_" + ts + "_" + std::to_string(n) + ".json");
    return cand;
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main(int argc, char **argv)
{
    try
    {
        const Options args = parseArgs(argc, argv);
        UiState ui;

        // Build detectors
        std::vector<DetectorEntry> detectors;
        auto params = cv::aruco::DetectorParameters::create();
        params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
        if (args.family == "auto")
        {
            for (const auto &[name, id] : kDictionaries)
                detectors.push_back({name, cv::aruco::getPredefinedDictionary(id), params});
        }
        else
        {
            auto it = kDictionaries.find(args.family);
            if (it == kDictionaries.end())
                throw std::runtime_error("Unsupported family: " + args.family);
            detectors.push_back({it->first, cv::aruco::getPredefinedDictionary(it->second), params});
        }

        cv::VideoCapture camera = openCamera(args.camera, args.search_limit);
        fs::path cal_dir = determineCalibrationDir(args);
        std::optional<RectificationCalibration> calibration;
        fs::path latest_cal_file;

        // Load calibration
        if (!args.load_calibration.empty())
        {
            if (!tryLoadCalibrationFile(args.load_calibration, args, calibration, latest_cal_file, ui))
                throw std::runtime_error(ui.status_text);
        }
        else if (args.auto_load_calibration)
        {
            auto files = listCalibrationFiles(cal_dir);
            if (!files.empty())
            {
                if (!tryLoadCalibrationFile(files.front(), args, calibration, latest_cal_file, ui))
                    throw std::runtime_error(ui.status_text);
            }
        }

        std::cout << "Press q to quit.\n";
        std::cout << "Use Lock [L], Clear [C], Save [S], Load [O] buttons or keys.\n";
        ui.status_text = calibration ? "Calibration ready" : "Show an AprilTag and lock calibration";

        cv::namedWindow(kWindowName);
        cv::setMouseCallback(kWindowName, onMouse);
        auto last_frame_time = std::chrono::steady_clock::now();

        while (true)
        {
            cv::Mat raw_frame;
            if (!camera.read(raw_frame) || raw_frame.empty())
            {
                std::cerr << "Failed to read frame.\n";
                break;
            }

            auto now = std::chrono::steady_clock::now();
            const double dt = std::chrono::duration<double>(now - last_frame_time).count();
            last_frame_time = now;
            if (dt > 0.0)
            {
                const double fps = 1.0 / dt;
                ui.fps_ema = ui.fps_ema <= 0.0 ? fps : (ui.fps_ema * 0.85 + fps * 0.15);
            }

            cv::Mat prepared = prepareCaptureFrame(raw_frame);
            cv::Mat clean;
            if (args.flip_horizontal)
                cv::flip(prepared, clean, 1);
            else
                clean = prepared;

            // Scale the small capture frame up to fill the fixed panel container
            cv::Mat display;
            cv::resize(clean, display, {kPanelW, kPanelH}, 0, 0, cv::INTER_NEAREST);
            const float dsx = (float)kPanelW / clean.cols;
            const float dsy = (float)kPanelH / clean.rows;

            // Determine active calibration pointer
            const RectificationCalibration *active =
                (args.freeze_calibration && calibration) ? &*calibration : nullptr;

            // Live detection (only when not frozen)
            std::vector<Detection> live;
            if (!active)
            {
                cv::Mat gray;
                cv::cvtColor(clean, gray, cv::COLOR_BGR2GRAY);
                live = detectTags(gray, detectors);
                // Auto-lock
                if (!live.empty() && args.auto_lock && args.freeze_calibration)
                {
                    const auto &f0 = live[0];
                    calibration = buildCalibration(clean, f0.family_name, f0.marker_id, f0.corners, args.flip_horizontal);
                    active = &*calibration;
                    if (args.auto_save_calibration || !args.save_calibration.empty())
                    {
                        latest_cal_file = buildUniquePath(args.save_calibration, args.calibration_file);
                        saveCalibration(*calibration, latest_cal_file);
                        refreshCalibrationPicker(ui, cal_dir);
                        ui.selected_calibration = 0;
                        std::cout << "Saved calibration to " << latest_cal_file << "\n";
                    }
                    ui.status_text = "Locked on " + f0.family_name + " id=" + std::to_string(f0.marker_id);
                    std::cout << "Locked on " << f0.family_name << " id=" << f0.marker_id << "\n";
                }
            }

            // Overlays — corners are in clean (160×120) space; scale to display (kPanelW×kPanelH)
            const std::vector<cv::Point2f> *plot_corners = nullptr;
            if (active)
            {
                plot_corners = &active->display_corners;
                auto sc = scaleCorners(active->display_corners, dsx, dsy);
                drawDetectionOverlay(display, active->family_name, active->marker_id, sc);
                cv::putText(display, "Calibration locked", {20, 35},
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 140, 0), 2, cv::LINE_AA);
            }
            else
            {
                for (const auto &d : live)
                {
                    auto sc = scaleCorners(d.corners, dsx, dsy);
                    drawDetectionOverlay(display, d.family_name, d.marker_id, sc);
                }
                if (live.empty())
                    cv::putText(display, "No AprilTag detected", {20, 35},
                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                else
                    ui.status_text = "Live detection ready - lock to freeze calibration";
                if (!live.empty())
                    plot_corners = &live[0].corners;
            }
            drawControls(display, calibration.has_value(), !live.empty());

            // Build combined 3-panel view — all panels share the same fixed kPanelW×kPanelH container
            cv::Mat warped_preview;
            cv::Mat panels[3] = {
                display,
                buildCornerPlot(plot_corners, kPanelW, kPanelH),
                buildRectifiedView(clean, active, kPanelW, kPanelH, &warped_preview)};
            cv::Mat combined;
            cv::hconcat(panels, 3, combined);
            cv::Mat status_bar = buildStatusBar(combined.cols, ui, latest_cal_file,
                                                calibration.has_value(), ui.load_picker_open);
            cv::Mat canvas;
            cv::vconcat(combined, status_bar, canvas);
            if (ui.load_picker_open)
                drawLoadPicker(canvas, ui, latest_cal_file);
            cv::imshow(kWindowName, canvas);

            // Input
            DisplayInput di;
            int kc = cv::waitKey(1);
            if (kc >= 0)
                di.key = kc & 0xFF;
            di.key = normalizeKey(di.key);
            di.click = g_pending_click;
            g_pending_click.reset();

            if (di.key == 'q')
                break;

            bool locked = calibration.has_value();
            bool can_lock = !live.empty();

            if (ui.load_picker_open)
            {
                refreshCalibrationPicker(ui, cal_dir);
                LoadPickerLayout layout = buildLoadPickerLayout(canvas.cols, canvas.rows, ui.calibration_files.size());

                if ((di.key == 'w' || di.key == 'k') && !ui.calibration_files.empty())
                {
                    ui.selected_calibration = std::max(0, ui.selected_calibration - 1);
                }
                else if ((di.key == 's' || di.key == 'j') && !ui.calibration_files.empty())
                {
                    ui.selected_calibration = std::min((int)ui.calibration_files.size() - 1,
                                                       ui.selected_calibration + 1);
                }
                else if (di.key >= '1' && di.key <= '9')
                {
                    int idx = ui.picker_scroll + (di.key - '1');
                    if (idx >= 0 && idx < (int)ui.calibration_files.size())
                        ui.selected_calibration = idx;
                }

                if (!ui.calibration_files.empty())
                {
                    const int max_scroll = std::max(0, (int)ui.calibration_files.size() - kPickerMaxVisible);
                    if (ui.selected_calibration < ui.picker_scroll)
                    {
                        ui.picker_scroll = ui.selected_calibration;
                    }
                    else if (ui.selected_calibration >= ui.picker_scroll + kPickerMaxVisible)
                    {
                        ui.picker_scroll = ui.selected_calibration - kPickerMaxVisible + 1;
                    }
                    ui.picker_scroll = std::clamp(ui.picker_scroll, 0, max_scroll);
                }

                if (di.click)
                {
                    for (size_t i = 0; i < layout.item_rects.size(); ++i)
                    {
                        if (layout.item_rects[i].contains(*di.click))
                        {
                            ui.selected_calibration = ui.picker_scroll + static_cast<int>(i);
                        }
                    }
                    if (layout.cancel_rect.contains(*di.click) || !layout.panel_rect.contains(*di.click))
                    {
                        ui.load_picker_open = false;
                        ui.status_text = "Load cancelled";
                        continue;
                    }
                    if (layout.load_rect.contains(*di.click) && !ui.calibration_files.empty())
                    {
                        tryLoadCalibrationFile(ui.calibration_files[ui.selected_calibration], args,
                                               calibration, latest_cal_file, ui);
                        ui.load_picker_open = false;
                        continue;
                    }
                }

                if (shouldCancel(di))
                {
                    ui.load_picker_open = false;
                    ui.status_text = "Load cancelled";
                    continue;
                }
                if ((shouldConfirm(di) || di.key == 'o') && !ui.calibration_files.empty())
                {
                    tryLoadCalibrationFile(ui.calibration_files[ui.selected_calibration], args,
                                           calibration, latest_cal_file, ui);
                    ui.load_picker_open = false;
                }
                continue;
            }

            if (di.key == 27)
                break;

            if (shouldClear(di, locked))
            {
                calibration.reset();
                latest_cal_file.clear();
                ui.status_text = "Calibration cleared";
                std::cout << "Cleared calibration.\n";
            }
            if (shouldLock(di, can_lock) && args.freeze_calibration && !live.empty())
            {
                const auto &f0 = live[0];
                calibration = buildCalibration(clean, f0.family_name, f0.marker_id, f0.corners, args.flip_horizontal);
                if (args.auto_save_calibration || !args.save_calibration.empty())
                {
                    latest_cal_file = buildUniquePath(args.save_calibration, args.calibration_file);
                    saveCalibration(*calibration, latest_cal_file);
                    refreshCalibrationPicker(ui, cal_dir);
                    ui.selected_calibration = 0;
                    std::cout << "Saved calibration to " << latest_cal_file << "\n";
                }
                ui.status_text = "Locked on " + f0.family_name + " id=" + std::to_string(f0.marker_id);
                std::cout << "Locked on " << f0.family_name << " id=" << f0.marker_id << "\n";
            }
            if (shouldSave(di, locked) && calibration)
            {
                latest_cal_file = buildUniquePath(args.save_calibration, args.calibration_file);
                saveCalibration(*calibration, latest_cal_file);
                refreshCalibrationPicker(ui, cal_dir);
                ui.selected_calibration = 0;
                ui.status_text = "Saved " + calibrationLabel(latest_cal_file);
                std::cout << "Saved calibration to " << latest_cal_file << "\n";
            }
            if (shouldLoad(di))
            {
                refreshCalibrationPicker(ui, cal_dir);
                ui.load_picker_open = true;
                if (!latest_cal_file.empty())
                {
                    auto it = std::find(ui.calibration_files.begin(), ui.calibration_files.end(), latest_cal_file);
                    if (it != ui.calibration_files.end())
                    {
                        ui.selected_calibration = static_cast<int>(std::distance(ui.calibration_files.begin(), it));
                    }
                }
                ui.picker_scroll = (ui.selected_calibration / kPickerMaxVisible) * kPickerMaxVisible;
                ui.status_text = ui.calibration_files.empty()
                                     ? "No calibration files found"
                                     : "Select a calibration version to load";
            }
        }

        camera.release();
        cv::destroyAllWindows();
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
