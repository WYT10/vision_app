#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#if __has_include(<opencv2/aruco.hpp>)
#include <opencv2/aruco.hpp>
#define VISION_APP_HAS_ARUCO 1
#else
#define VISION_APP_HAS_ARUCO 0
#endif

namespace vision_app {

struct AprilTagConfig {
    std::string family = "tag36h11";
    int target_id = 0;
    bool require_target_id = true;
    int lock_frames = 8;
    double max_center_jitter_px = 3.0;
    double max_corner_jitter_px = 4.0;
    double min_quad_area_ratio = 0.0025;
    int threads = 2;
    float decimate = 1.0f;
    float blur_sigma = 0.0f;
    bool refine_edges = true;
    bool show_family = true;
};

struct AprilTagDetection {
    bool found = false;
    std::string family = "";
    int id = -1;
    double area = 0.0;
    double decision_margin = 0.0;
    int hamming = 0;
    std::array<cv::Point2f, 4> corners{};
    cv::Point2f center{};
};

struct HomographyLock {
    bool valid = false;
    std::string family = "";
    int id = -1;
    cv::Size warp_size{1280, 720};
    cv::Mat H;     // src->dst
    cv::Mat Hinv;  // dst->src
    std::array<cv::Point2f, 4> locked_corners{};
};

struct RoiRatio {
    double x = 0.0;
    double y = 0.0;
    double w = 0.0;
    double h = 0.0;
};

struct RoiSet {
    RoiRatio red_roi{0.05, 0.10, 0.20, 0.20};
    RoiRatio image_roi{0.30, 0.10, 0.50, 0.60};
};

static bool clamp_and_validate_roi(RoiRatio& roi) {
    auto clamp01 = [](double v) { return std::max(0.0, std::min(1.0, v)); };
    roi.x = clamp01(roi.x);
    roi.y = clamp01(roi.y);
    roi.w = clamp01(roi.w);
    roi.h = clamp01(roi.h);
    if (roi.x + roi.w > 1.0) roi.w = 1.0 - roi.x;
    if (roi.y + roi.h > 1.0) roi.h = 1.0 - roi.y;
    return roi.w > 0.0 && roi.h > 0.0;
}

static cv::Rect ratio_to_rect(const RoiRatio& roi, const cv::Size& image_size) {
    RoiRatio r = roi;
    clamp_and_validate_roi(r);
    int x = static_cast<int>(std::round(r.x * image_size.width));
    int y = static_cast<int>(std::round(r.y * image_size.height));
    int w = std::max(1, static_cast<int>(std::round(r.w * image_size.width)));
    int h = std::max(1, static_cast<int>(std::round(r.h * image_size.height)));
    x = std::max(0, std::min(x, image_size.width - 1));
    y = std::max(0, std::min(y, image_size.height - 1));
    if (x + w > image_size.width) w = image_size.width - x;
    if (y + h > image_size.height) h = image_size.height - y;
    return cv::Rect(x, y, std::max(1, w), std::max(1, h));
}

static std::array<cv::Point2f, 4> make_warp_quad(const cv::Size& size) {
    return {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(size.width - 1), 0.0f),
        cv::Point2f(static_cast<float>(size.width - 1), static_cast<float>(size.height - 1)),
        cv::Point2f(0.0f, static_cast<float>(size.height - 1))
    };
}

// Normalize user-supplied family name.
// Accepts short forms ("16", "25", "36") and full names ("tag16h5", …).
// The special value "auto" tries all families and picks the largest detected tag.
static std::string normalize_tag_family(const std::string& s) {
    if (s == "auto")                                   return "auto";
    if (s == "16" || s == "16h5"  || s == "tag16h5")  return "tag16h5";
    if (s == "25" || s == "25h9"  || s == "tag25h9")  return "tag25h9";
    if (s == "36h10"              || s == "tag36h10") return "tag36h10";
    if (s == "36" || s == "36h11" || s == "tag36h11") return "tag36h11";
    return s; // pass through unknown values as-is
}

#if VISION_APP_HAS_ARUCO
static int april_family_to_dict(const std::string& family) {
    if (family == "tag16h5")  return cv::aruco::DICT_APRILTAG_16h5;
    if (family == "tag25h9")  return cv::aruco::DICT_APRILTAG_25h9;
    if (family == "tag36h10") return cv::aruco::DICT_APRILTAG_36h10;
    return cv::aruco::DICT_APRILTAG_36h11;
}
#endif

static bool detect_apriltag_best(const cv::Mat& bgr_or_gray,
                                 const AprilTagConfig& cfg,
                                 AprilTagDetection& out,
                                 std::string& err) {
    out = {};
#if VISION_APP_HAS_ARUCO
    cv::Mat gray;
    if (bgr_or_gray.channels() == 1) gray = bgr_or_gray;
    else cv::cvtColor(bgr_or_gray, gray, cv::COLOR_BGR2GRAY);

    // Helper lambda: detect with one specific dictionary and return the best hit.
    auto detect_with_dict = [&](int dict_id, const std::string& fname) -> AprilTagDetection {
        AprilTagDetection d;
        auto dict   = cv::aruco::getPredefinedDictionary(dict_id);
        auto params = cv::aruco::DetectorParameters();
#if CV_VERSION_MAJOR >= 4
        params.cornerRefinementMethod = cfg.refine_edges
            ? cv::aruco::CORNER_REFINE_SUBPIX : cv::aruco::CORNER_REFINE_NONE;
#endif
        params.aprilTagQuadDecimate = cfg.decimate;
        params.aprilTagQuadSigma    = cfg.blur_sigma;
        params.useAruco3Detection   = false;

        cv::aruco::ArucoDetector detector(dict, params);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        detector.detectMarkers(gray, corners, ids);
        if (ids.empty()) return d;

        int best_i = -1;
        double best_area = -1.0;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (cfg.require_target_id && ids[i] != cfg.target_id) continue;
            if (corners[i].size() != 4) continue;
            const double area = std::abs(cv::contourArea(corners[i]));
            if (area > best_area) { best_area = area; best_i = static_cast<int>(i); }
        }
        if (best_i < 0) return d;

        const auto& c = corners[best_i];
        d.found   = true;
        d.family  = fname;
        d.id      = ids[best_i];
        d.area    = std::abs(cv::contourArea(c));
        d.corners = {c[0], c[1], c[2], c[3]};
        d.center  = 0.25f * (c[0] + c[1] + c[2] + c[3]);
        return d;
    };

    // "auto": scan all four families; keep the largest-area tag found.
    if (cfg.family == "auto") {
        const std::pair<int, const char*> kFamilies[] = {
            {cv::aruco::DICT_APRILTAG_36h11, "tag36h11"},
            {cv::aruco::DICT_APRILTAG_25h9,  "tag25h9"},
            {cv::aruco::DICT_APRILTAG_16h5,  "tag16h5"},
            {cv::aruco::DICT_APRILTAG_36h10, "tag36h10"},
        };
        for (const auto& kv : kFamilies) {
            AprilTagDetection d = detect_with_dict(kv.first, kv.second);
            if (d.found && (!out.found || d.area > out.area)) out = d;
        }
        return true;
    }

    out = detect_with_dict(april_family_to_dict(cfg.family), cfg.family);
    return true;
#else
    (void)bgr_or_gray; (void)cfg;
    err = "AprilTag support unavailable. Install OpenCV aruco/contrib or swap in an AprilTag backend.";
    return false;
#endif
}

class TagLocker {
public:
    explicit TagLocker(const AprilTagConfig& cfg) : cfg_(cfg) {}

    void reset() {
        hist_.clear();
        locked_ = {};
    }

    bool update(const AprilTagDetection& det, const cv::Size& image_size) {
        if (!det.found) {
            reset();
            return false;
        }

        const double area_ratio = det.area / std::max(1.0, static_cast<double>(image_size.area()));
        if (area_ratio < cfg_.min_quad_area_ratio) {
            reset();
            return false;
        }

        if (!hist_.empty()) {
            if (hist_.back().id != det.id || hist_.back().family != det.family) {
                reset();
            }
        }

        hist_.push_back(det);
        while (static_cast<int>(hist_.size()) > cfg_.lock_frames) hist_.pop_front();
        if (static_cast<int>(hist_.size()) < cfg_.lock_frames) return false;

        const cv::Point2f ref_center = hist_.front().center;
        const auto& ref_corners = hist_.front().corners;
        for (const auto& d : hist_) {
            const double dc = cv::norm(d.center - ref_center);
            if (dc > cfg_.max_center_jitter_px) {
                reset();
                return false;
            }
            for (int i = 0; i < 4; ++i) {
                const double dj = cv::norm(d.corners[i] - ref_corners[i]);
                if (dj > cfg_.max_corner_jitter_px) {
                    reset();
                    return false;
                }
            }
        }

        locked_.found = true;
        locked_.family = hist_.back().family;
        locked_.id = hist_.back().id;
        locked_.corners = hist_.back().corners;
        locked_.center = hist_.back().center;
        locked_.area = hist_.back().area;
        return true;
    }

    bool locked() const { return locked_.found; }
    AprilTagDetection locked_det() const { return locked_; }

private:
    AprilTagConfig cfg_;
    std::deque<AprilTagDetection> hist_;
    AprilTagDetection locked_{};
};

static bool compute_homography_from_tag_quad(const AprilTagDetection& det,
                                             const cv::Size& warp_size,
                                             HomographyLock& out) {
    if (!det.found || warp_size.width <= 0 || warp_size.height <= 0) return false;
    std::vector<cv::Point2f> src(4), dst(4);
    for (int i = 0; i < 4; ++i) src[i] = det.corners[i];
    const auto quad = make_warp_quad(warp_size);
    for (int i = 0; i < 4; ++i) dst[i] = quad[i];
    cv::Mat H = cv::getPerspectiveTransform(src, dst);
    if (H.empty()) return false;
    out.valid = true;
    out.family = det.family;
    out.id = det.id;
    out.warp_size = warp_size;
    out.H = H.clone();
    out.Hinv = H.inv();
    out.locked_corners = det.corners;
    return !out.H.empty() && !out.Hinv.empty();
}

static bool warp_full_frame(const cv::Mat& src, cv::Mat& dst, const HomographyLock& lock, int interp = cv::INTER_LINEAR) {
    if (!lock.valid || lock.H.empty() || src.empty()) return false;
    cv::warpPerspective(src, dst, lock.H, lock.warp_size, interp, cv::BORDER_CONSTANT);
    return !dst.empty();
}

static void draw_detection_overlay(cv::Mat& frame, const AprilTagDetection& det, bool locked) {
    if (!det.found) {
        cv::putText(frame, "AprilTag: not found", {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2);
        return;
    }
    const cv::Scalar color = locked ? cv::Scalar(0,255,0) : cv::Scalar(0,255,255);
    for (int i = 0; i < 4; ++i) {
        cv::line(frame, det.corners[i], det.corners[(i + 1) % 4], color, 2);
    }
    cv::circle(frame, det.center, 4, {255, 0, 255}, -1);
    std::ostringstream oss;
    oss << "Tag " << det.family << " id=" << det.id;
    if (locked) oss << " [LOCKED]";
    cv::putText(frame, oss.str(), {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
}

static void draw_roi_overlay(cv::Mat& warped, const RoiSet& rois, const HomographyLock& lock) {
    const cv::Rect rr = ratio_to_rect(rois.red_roi, warped.size());
    const cv::Rect ir = ratio_to_rect(rois.image_roi, warped.size());
    cv::rectangle(warped, rr, {0,0,255}, 2);
    cv::putText(warped, "red_roi", rr.tl() + cv::Point(4, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,255}, 2);
    cv::rectangle(warped, ir, {255,0,0}, 2);
    cv::putText(warped, "image_roi", ir.tl() + cv::Point(4, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,0,0}, 2);
    std::ostringstream oss;
    oss << "Warp lock: " << lock.family << " id=" << lock.id;
    cv::putText(warped, oss.str(), {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.75, {0,255,0}, 2);
}

static bool save_homography_json(const std::string& path, const HomographyLock& lock) {
    if (!lock.valid || lock.H.empty()) return false;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << std::fixed << std::setprecision(10);
    out << "{\n";
    out << "  \"family\": \"" << lock.family << "\",\n";
    out << "  \"id\": " << lock.id << ",\n";
    out << "  \"warp_width\": " << lock.warp_size.width << ",\n";
    out << "  \"warp_height\": " << lock.warp_size.height << ",\n";
    out << "  \"H\": [\n";
    for (int r = 0; r < 3; ++r) {
        out << "    [";
        for (int c = 0; c < 3; ++c) {
            out << lock.H.at<double>(r, c);
            if (c < 2) out << ", ";
        }
        out << "]" << (r < 2 ? "," : "") << "\n";
    }
    out << "  ]\n}";
    return true;
}

static bool save_rois_json(const std::string& path, const RoiSet& rois) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    if (!out.is_open()) return false;
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"red_roi\": {\"x\": " << rois.red_roi.x << ", \"y\": " << rois.red_roi.y << ", \"w\": " << rois.red_roi.w << ", \"h\": " << rois.red_roi.h << "},\n";
    out << "  \"image_roi\": {\"x\": " << rois.image_roi.x << ", \"y\": " << rois.image_roi.y << ", \"w\": " << rois.image_roi.w << ", \"h\": " << rois.image_roi.h << "}\n";
    out << "}\n";
    return true;
}

static bool extract_scalar_after(const std::string& s, size_t start, double& value) {
    size_t p = start;
    while (p < s.size() && !(std::isdigit(static_cast<unsigned char>(s[p])) || s[p] == '-' || s[p] == '.')) ++p;
    if (p >= s.size()) return false;
    size_t e = p + 1;
    while (e < s.size() && (std::isdigit(static_cast<unsigned char>(s[e])) || s[e] == '.' || s[e] == '-' || s[e] == '+' || s[e] == 'e' || s[e] == 'E')) ++e;
    value = std::stod(s.substr(p, e - p));
    return true;
}

static bool extract_int_key(const std::string& s, const std::string& key, int& value) {
    const std::string token = "\"" + key + "\"";
    const size_t p = s.find(token);
    if (p == std::string::npos) return false;
    double tmp = 0.0;
    if (!extract_scalar_after(s, p + token.size(), tmp)) return false;
    value = static_cast<int>(std::llround(tmp));
    return true;
}

static bool extract_named_roi(const std::string& s, const std::string& key, RoiRatio& roi) {
    const std::string token = "\"" + key + "\"";
    size_t p = s.find(token);
    if (p == std::string::npos) return false;
    size_t block_end = s.find('}', p);
    if (block_end == std::string::npos) return false;
    const std::string block = s.substr(p, block_end - p + 1);

    auto get = [&](const std::string& subkey, double& outv) -> bool {
        const std::string t = "\"" + subkey + "\"";
        const size_t q = block.find(t);
        if (q == std::string::npos) return false;
        return extract_scalar_after(block, q + t.size(), outv);
    };

    if (!get("x", roi.x)) return false;
    if (!get("y", roi.y)) return false;
    if (!get("w", roi.w)) return false;
    if (!get("h", roi.h)) return false;
    return clamp_and_validate_roi(roi);
}

static bool load_rois_json(const std::string& path, RoiSet& rois) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();
    RoiSet tmp = rois;
    if (!extract_named_roi(s, "red_roi", tmp.red_roi)) return false;
    if (!extract_named_roi(s, "image_roi", tmp.image_roi)) return false;
    rois = tmp;
    return true;
}

static bool load_homography_json(const std::string& path, HomographyLock& lock) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();

    int w = 0, h = 0, id = -1;
    if (!extract_int_key(s, "warp_width", w)) return false;
    if (!extract_int_key(s, "warp_height", h)) return false;
    if (!extract_int_key(s, "id", id)) return false;

    std::vector<double> vals;
    vals.reserve(9);
    size_t p = s.find("\"H\"");
    if (p == std::string::npos) return false;
    while (p < s.size() && vals.size() < 9) {
        double v = 0.0;
        if (extract_scalar_after(s, p, v)) {
            vals.push_back(v);
            size_t next = p;
            while (next < s.size() && !(std::isdigit(static_cast<unsigned char>(s[next])) || s[next] == '-' || s[next] == '.')) ++next;
            while (next < s.size() && (std::isdigit(static_cast<unsigned char>(s[next])) || s[next] == '.' || s[next] == '-' || s[next] == '+' || s[next] == 'e' || s[next] == 'E')) ++next;
            p = next;
        } else {
            ++p;
        }
    }
    if (vals.size() < 9) return false;

    cv::Mat H(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) H.at<double>(i / 3, i % 3) = vals[i];

    lock.valid = true;
    lock.id = id;
    lock.warp_size = cv::Size(w, h);
    lock.H = H.clone();
    lock.Hinv = H.inv();
    return !lock.H.empty() && !lock.Hinv.empty();
}

static cv::Mat crop_roi_clone(const cv::Mat& warped, const RoiRatio& roi) {
    if (warped.empty()) return {};
    return warped(ratio_to_rect(roi, warped.size())).clone();
}

// ---------------------------------------------------------------------------
// Mouse-drag support for the warp-preview window.
// Register once with cv::setMouseCallback("vision_app_warp", on_warp_mouse, &data).
// Left-button drag  : moves the selected ROI.
// Right-button drag : resizes the selected ROI by dragging its bottom-right corner.
// ---------------------------------------------------------------------------
struct WarpMouseCbData {
    RoiSet*  rois         = nullptr;
    cv::Size warp_size    {};
    bool     lock_valid   = false;   // set to true once the homography is locked
    char*    selected_roi = nullptr; // pointer to the caller's '1'/'2' selector

    // internal drag state
    bool     is_dragging  = false;
    bool     is_resizing  = false;
    int      drag_roi     = 0;       // 1 = red_roi, 2 = image_roi
    cv::Point drag_start  {};
    RoiRatio roi_at_drag_start {};
};

static void on_warp_mouse(int event, int x, int y, int /*flags*/, void* userdata) {
    auto* d = static_cast<WarpMouseCbData*>(userdata);
    if (!d || !d->rois || !d->lock_valid || d->warp_size.area() == 0) return;

    const cv::Point pt(x, y);

    auto pick_roi = [&]() -> int {
        const cv::Rect rr = ratio_to_rect(d->rois->red_roi,   d->warp_size);
        const cv::Rect ir = ratio_to_rect(d->rois->image_roi, d->warp_size);
        if (rr.contains(pt)) return 1;
        if (ir.contains(pt)) return 2;
        return 0;
    };

    auto current_roi = [&]() -> RoiRatio& {
        return (d->drag_roi == 1) ? d->rois->red_roi : d->rois->image_roi;
    };

    if (event == cv::EVENT_LBUTTONDOWN) {
        d->drag_roi = pick_roi();
        if (d->drag_roi) {
            d->is_dragging        = true;
            d->drag_start         = pt;
            d->roi_at_drag_start  = (d->drag_roi == 1) ? d->rois->red_roi : d->rois->image_roi;
            if (d->selected_roi) *d->selected_roi = static_cast<char>('0' + d->drag_roi);
        }
    } else if (event == cv::EVENT_RBUTTONDOWN) {
        d->drag_roi = pick_roi();
        if (d->drag_roi) {
            d->is_resizing        = true;
            d->drag_start         = pt;
            d->roi_at_drag_start  = (d->drag_roi == 1) ? d->rois->red_roi : d->rois->image_roi;
            if (d->selected_roi) *d->selected_roi = static_cast<char>('0' + d->drag_roi);
        }
    } else if (event == cv::EVENT_MOUSEMOVE) {
        if (d->is_dragging && d->drag_roi) {
            const double dx = static_cast<double>(x - d->drag_start.x) / d->warp_size.width;
            const double dy = static_cast<double>(y - d->drag_start.y) / d->warp_size.height;
            RoiRatio& roi  = current_roi();
            roi.x = d->roi_at_drag_start.x + dx;
            roi.y = d->roi_at_drag_start.y + dy;
            roi.w = d->roi_at_drag_start.w;
            roi.h = d->roi_at_drag_start.h;
            clamp_and_validate_roi(roi);
        } else if (d->is_resizing && d->drag_roi) {
            // Right-drag moves the bottom-right corner, changing w and h.
            const double dx = static_cast<double>(x - d->drag_start.x) / d->warp_size.width;
            const double dy = static_cast<double>(y - d->drag_start.y) / d->warp_size.height;
            RoiRatio& roi  = current_roi();
            roi.x = d->roi_at_drag_start.x;
            roi.y = d->roi_at_drag_start.y;
            roi.w = d->roi_at_drag_start.w + dx;
            roi.h = d->roi_at_drag_start.h + dy;
            clamp_and_validate_roi(roi);
        }
    } else if (event == cv::EVENT_LBUTTONUP) {
        d->is_dragging = false;
        d->drag_roi    = 0;
    } else if (event == cv::EVENT_RBUTTONUP) {
        d->is_resizing = false;
        d->drag_roi    = 0;
    }
}

} // namespace vision_app
