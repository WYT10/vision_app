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
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>

#if __has_include(<opencv2/aruco.hpp>)
#include <opencv2/aruco.hpp>
#define VISION_APP_HAS_ARUCO 1
#else
#define VISION_APP_HAS_ARUCO 0
#endif

namespace vision_app {

struct AprilTagConfig {
    std::string family = "auto";          // auto | 16 | 25 | 36 | tag16h5 | tag25h9 | tag36h11
    int target_id = 0;
    bool require_target_id = true;
    bool manual_lock_only = false;         // when true, only Space/Enter can lock
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

struct WarpRemapCache {
    bool valid = false;
    bool fixed_point = true;
    cv::Size source_size{0, 0};
    cv::Size warp_size{0, 0};
    cv::Mat map1;
    cv::Mat map2;
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
    x = std::max(0, std::min(x, std::max(0, image_size.width - 1)));
    y = std::max(0, std::min(y, std::max(0, image_size.height - 1)));
    if (x + w > image_size.width) w = image_size.width - x;
    if (y + h > image_size.height) h = image_size.height - y;
    return cv::Rect(x, y, std::max(1, w), std::max(1, h));
}

static cv::Mat crop_roi_clone(const cv::Mat& img, const RoiRatio& roi) {
    if (img.empty()) return {};
    const cv::Rect r = ratio_to_rect(roi, img.size());
    return img(r).clone();
}

static std::array<cv::Point2f, 4> make_warp_quad(const cv::Size& size) {
    return {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(static_cast<float>(size.width - 1), 0.0f),
        cv::Point2f(static_cast<float>(size.width - 1), static_cast<float>(size.height - 1)),
        cv::Point2f(0.0f, static_cast<float>(size.height - 1))
    };
}

static std::string normalize_family_token(std::string family) {
    std::transform(family.begin(), family.end(), family.begin(), [](unsigned char c){ return static_cast<char>(std::tolower(c)); });
    if (family == "tag16h5" || family == "16" || family == "apriltag16" || family == "16h5") return "16";
    if (family == "tag25h9" || family == "25" || family == "apriltag25" || family == "25h9") return "25";
    if (family == "tag36h11" || family == "tag36h10" || family == "36" || family == "apriltag36" || family == "36h11" || family == "36h10") return "36";
    if (family == "auto" || family.empty()) return "auto";
    return family;
}

static std::vector<std::string> tag_family_search_order(const std::string& family_token) {
    const std::string f = normalize_family_token(family_token);
    if (f == "16") return {"16"};
    if (f == "25") return {"25"};
    if (f == "36") return {"36"};
    return {"36", "25", "16"};
}

static std::string family_label(const std::string& token) {
    const std::string f = normalize_family_token(token);
    if (f == "16") return "tag16h5";
    if (f == "25") return "tag25h9";
    if (f == "36") return "tag36h11";
    return "unknown";
}

#if VISION_APP_HAS_ARUCO
static int family_token_to_dict(const std::string& token) {
    const std::string f = normalize_family_token(token);
    if (f == "16") return cv::aruco::DICT_APRILTAG_16h5;
    if (f == "25") return cv::aruco::DICT_APRILTAG_25h9;
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

    auto search_order = tag_family_search_order(cfg.family);
    double best_score = -1.0;

    for (const auto& fam : search_order) {
        auto dict = cv::aruco::getPredefinedDictionary(family_token_to_dict(fam));
        auto params = cv::aruco::DetectorParameters();
        params.cornerRefinementMethod = cfg.refine_edges ? cv::aruco::CORNER_REFINE_SUBPIX : cv::aruco::CORNER_REFINE_NONE;
        params.aprilTagQuadDecimate = cfg.decimate;
        params.aprilTagQuadSigma = cfg.blur_sigma;
        params.useAruco3Detection = false;

        cv::aruco::ArucoDetector detector(dict, params);
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        detector.detectMarkers(gray, corners, ids);

        for (size_t i = 0; i < ids.size(); ++i) {
            if (cfg.require_target_id && ids[i] != cfg.target_id) continue;
            if (corners[i].size() != 4) continue;
            const double area = std::abs(cv::contourArea(corners[i]));
            const double score = area;
            if (score <= best_score) continue;
            best_score = score;
            out.found = true;
            out.family = family_label(fam);
            out.id = ids[i];
            out.area = area;
            out.corners = {corners[i][0], corners[i][1], corners[i][2], corners[i][3]};
            out.center = 0.25f * (corners[i][0] + corners[i][1] + corners[i][2] + corners[i][3]);
        }
    }
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
        last_candidate_ = {};
    }

    bool update(const AprilTagDetection& det, const cv::Size& image_size, bool allow_lock = true) {
        last_candidate_ = det;
        if (!det.found) {
            if (!allow_lock) return false;
            reset();
            return false;
        }

        const double area_ratio = det.area / std::max(1.0, static_cast<double>(image_size.area()));
        if (area_ratio < cfg_.min_quad_area_ratio) {
            if (!allow_lock) return false;
            reset();
            return false;
        }

        if (!hist_.empty()) {
            if (hist_.back().id != det.id || hist_.back().family != det.family) {
                hist_.clear();
            }
        }

        hist_.push_back(det);
        while (static_cast<int>(hist_.size()) > cfg_.lock_frames) hist_.pop_front();
        if (!allow_lock || static_cast<int>(hist_.size()) < cfg_.lock_frames) return false;

        const cv::Point2f ref_center = hist_.front().center;
        const auto& ref_corners = hist_.front().corners;
        for (const auto& d : hist_) {
            const double dc = cv::norm(d.center - ref_center);
            if (dc > cfg_.max_center_jitter_px) {
                hist_.clear();
                return false;
            }
            for (int i = 0; i < 4; ++i) {
                const double dj = cv::norm(d.corners[i] - ref_corners[i]);
                if (dj > cfg_.max_corner_jitter_px) {
                    hist_.clear();
                    return false;
                }
            }
        }

        locked_ = hist_.back();
        locked_.found = true;
        return true;
    }

    bool force_lock(const AprilTagDetection& det, const cv::Size& image_size) {
        if (!det.found) return false;
        const double area_ratio = det.area / std::max(1.0, static_cast<double>(image_size.area()));
        if (area_ratio < cfg_.min_quad_area_ratio) return false;
        locked_ = det;
        locked_.found = true;
        return true;
    }

    bool locked() const { return locked_.found; }
    AprilTagDetection locked_det() const { return locked_; }
    AprilTagDetection last_candidate() const { return last_candidate_; }
    int history_size() const { return static_cast<int>(hist_.size()); }

private:
    AprilTagConfig cfg_;
    std::deque<AprilTagDetection> hist_;
    AprilTagDetection locked_{};
    AprilTagDetection last_candidate_{};
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


static bool build_warp_remap_cache(const HomographyLock& lock,
                                   const cv::Size& source_size,
                                   WarpRemapCache& cache,
                                   bool fixed_point = true) {
    if (!lock.valid || lock.Hinv.empty()) return false;
    if (source_size.width <= 0 || source_size.height <= 0) return false;
    if (lock.warp_size.width <= 0 || lock.warp_size.height <= 0) return false;

    cv::Mat map_x(lock.warp_size, CV_32FC1);
    cv::Mat map_y(lock.warp_size, CV_32FC1);
    cv::Matx33d M{};
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            M(r, c) = lock.Hinv.at<double>(r, c);
        }
    }

    for (int y = 0; y < lock.warp_size.height; ++y) {
        float* mx = map_x.ptr<float>(y);
        float* my = map_y.ptr<float>(y);
        for (int x = 0; x < lock.warp_size.width; ++x) {
            const double xx = M(0,0) * x + M(0,1) * y + M(0,2);
            const double yy = M(1,0) * x + M(1,1) * y + M(1,2);
            const double ww = M(2,0) * x + M(2,1) * y + M(2,2);
            if (std::abs(ww) < 1e-12) {
                mx[x] = -1.0f;
                my[x] = -1.0f;
            } else {
                mx[x] = static_cast<float>(xx / ww);
                my[x] = static_cast<float>(yy / ww);
            }
        }
    }

    cache = {};
    cache.valid = true;
    cache.fixed_point = fixed_point;
    cache.source_size = source_size;
    cache.warp_size = lock.warp_size;
    if (fixed_point) {
        cv::convertMaps(map_x, map_y, cache.map1, cache.map2, CV_16SC2);
    } else {
        cache.map1 = map_x;
        cache.map2 = map_y;
    }
    return !cache.map1.empty();
}

static bool remap_cache_matches(const WarpRemapCache& cache,
                                const cv::Size& source_size,
                                const cv::Size& warp_size) {
    return cache.valid && cache.source_size == source_size && cache.warp_size == warp_size && !cache.map1.empty();
}

static bool warp_full_frame_cached(const cv::Mat& src,
                                   cv::Mat& dst,
                                   const WarpRemapCache& cache,
                                   int interp = cv::INTER_LINEAR) {
    if (src.empty() || !cache.valid || cache.map1.empty()) return false;
    if (src.size() != cache.source_size) return false;
    cv::remap(src, dst, cache.map1, cache.map2, interp, cv::BORDER_CONSTANT);
    return !dst.empty();
}

static bool save_warp_remap_cache(const std::string& path, const WarpRemapCache& cache) {
    if (!cache.valid || cache.map1.empty()) return false;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "valid" << true;
    fs << "fixed_point" << cache.fixed_point;
    fs << "source_width" << cache.source_size.width;
    fs << "source_height" << cache.source_size.height;
    fs << "warp_width" << cache.warp_size.width;
    fs << "warp_height" << cache.warp_size.height;
    fs << "map1" << cache.map1;
    fs << "map2" << cache.map2;
    fs.release();
    return true;
}

static bool load_warp_remap_cache(const std::string& path, WarpRemapCache& cache) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    int sw = 0, sh = 0, ww = 0, wh = 0;
    int fp = 1;
    cv::Mat map1, map2;
    fs["source_width"] >> sw;
    fs["source_height"] >> sh;
    fs["warp_width"] >> ww;
    fs["warp_height"] >> wh;
    fs["fixed_point"] >> fp;
    fs["map1"] >> map1;
    fs["map2"] >> map2;
    fs.release();
    if (sw <= 0 || sh <= 0 || ww <= 0 || wh <= 0 || map1.empty()) return false;
    cache = {};
    cache.valid = true;
    cache.fixed_point = (fp != 0);
    cache.source_size = cv::Size(sw, sh);
    cache.warp_size = cv::Size(ww, wh);
    cache.map1 = map1;
    cache.map2 = map2;
    return true;
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
    oss << det.family << " id=" << det.id;
    if (locked) oss << " [LOCKED]";
    cv::putText(frame, oss.str(), {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
}

static void draw_roi_overlay(cv::Mat& warped, const RoiSet& rois, const HomographyLock& lock, char selected_roi = '1') {
    const cv::Rect rr = ratio_to_rect(rois.red_roi, warped.size());
    const cv::Rect ir = ratio_to_rect(rois.image_roi, warped.size());
    const int red_thick = (selected_roi == '1') ? 3 : 2;
    const int img_thick = (selected_roi == '2') ? 3 : 2;
    cv::rectangle(warped, rr, {0,0,255}, red_thick);
    cv::putText(warped, "red_roi", rr.tl() + cv::Point(4, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,255}, 2);
    cv::rectangle(warped, ir, {255,0,0}, img_thick);
    cv::putText(warped, "image_roi", ir.tl() + cv::Point(4, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, {255,0,0}, 2);
    std::ostringstream oss;
    oss << "Warp lock: " << lock.family << " id=" << lock.id;
    cv::putText(warped, oss.str(), {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.75, {0,255,0}, 2);
}

static bool save_homography_json(const std::string& path, const HomographyLock& lock) {
    if (!lock.valid || lock.H.empty()) return false;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) return false;
    out << std::fixed << std::setprecision(10);
    out << "{\n";
    out << "  \"valid\": true,\n";
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
        out << "]";
        if (r < 2) out << ',';
        out << "\n";
    }
    out << "  ]\n";
    out << "}\n";
    return true;
}

static bool save_rois_json(const std::string& path, const RoiSet& rois) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::trunc);
    if (!out.is_open()) return false;
    out << std::fixed << std::setprecision(6);
    out << "{\n";
    out << "  \"red_roi\": {\"x\": " << rois.red_roi.x << ", \"y\": " << rois.red_roi.y << ", \"w\": " << rois.red_roi.w << ", \"h\": " << rois.red_roi.h << "},\n";
    out << "  \"image_roi\": {\"x\": " << rois.image_roi.x << ", \"y\": " << rois.image_roi.y << ", \"w\": " << rois.image_roi.w << ", \"h\": " << rois.image_roi.h << "}\n";
    out << "}\n";
    return true;
}

static bool extract_double_after(const std::string& s, size_t start, double& value) {
    size_t p = start;
    while (p < s.size() && !(std::isdigit(static_cast<unsigned char>(s[p])) || s[p] == '-' || s[p] == '.')) ++p;
    if (p >= s.size()) return false;
    size_t e = p + 1;
    while (e < s.size() && (std::isdigit(static_cast<unsigned char>(s[e])) || s[e] == '.' || s[e] == '-' || s[e] == 'e' || s[e] == 'E' || s[e] == '+')) ++e;
    value = std::stod(s.substr(p, e - p));
    return true;
}

static bool extract_named_double(const std::string& s, const std::string& section, const std::string& key, double& value) {
    size_t sec = s.find(section);
    if (sec == std::string::npos) return false;
    size_t k = s.find(key, sec);
    if (k == std::string::npos) return false;
    return extract_double_after(s, k + key.size(), value);
}

[[maybe_unused]] static bool extract_named_int(const std::string& s, const std::string& key, int& value) {
    double d = 0.0;
    if (!extract_named_double(s, key, key, d)) {
        size_t pos = s.find(key);
        if (pos == std::string::npos) return false;
        if (!extract_double_after(s, pos + key.size(), d)) return false;
    }
    value = static_cast<int>(std::lround(d));
    return true;
}

static bool load_homography_json(const std::string& path, HomographyLock& lock) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();

    int w = 0, h = 0, id = -1;
    size_t pw = s.find("\"warp_width\"");
    size_t ph = s.find("\"warp_height\"");
    size_t pi = s.find("\"id\"");
    if (pw == std::string::npos || ph == std::string::npos || pi == std::string::npos) return false;
    double tmp = 0.0;
    if (!extract_double_after(s, pw, tmp)) {
        return false;
    }
    w = static_cast<int>(std::lround(tmp));
    if (!extract_double_after(s, ph, tmp)) {
        return false;
    }
    h = static_cast<int>(std::lround(tmp));
    if (!extract_double_after(s, pi, tmp)) {
        return false;
    }
    id = static_cast<int>(std::lround(tmp));

    size_t pf = s.find("\"family\"");
    std::string family;
    if (pf != std::string::npos) {
        size_t colon = s.find(':', pf);
        size_t q1 = (colon == std::string::npos) ? std::string::npos : s.find('"', colon + 1);
        size_t q2 = (q1 == std::string::npos) ? std::string::npos : s.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
            family = s.substr(q1 + 1, q2 - q1 - 1);
        }
    }

    std::vector<double> vals;
    vals.reserve(9);
    size_t p = s.find("\"H\"");
    if (p == std::string::npos) return false;
    for (; p < s.size() && vals.size() < 9; ++p) {
        if (std::isdigit(static_cast<unsigned char>(s[p])) || s[p] == '-' || s[p] == '.') {
            size_t e = p + 1;
            while (e < s.size() && (std::isdigit(static_cast<unsigned char>(s[e])) || s[e] == '.' || s[e] == '-' || s[e] == 'e' || s[e] == 'E' || s[e] == '+')) ++e;
            vals.push_back(std::stod(s.substr(p, e - p)));
            p = e - 1;
        }
    }
    if (vals.size() != 9) return false;

    cv::Mat H(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) H.at<double>(i / 3, i % 3) = vals[i];
    lock.valid = true;
    lock.family = family;
    lock.id = id;
    lock.warp_size = cv::Size(w, h);
    lock.H = H.clone();
    lock.Hinv = H.inv();
    return true;
}

static bool load_rois_json(const std::string& path, RoiSet& rois) {
    std::ifstream in(path);
    if (!in.is_open()) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string s = ss.str();

    RoiSet tmp = rois;
    if (!extract_named_double(s, "\"red_roi\"", "\"x\"", tmp.red_roi.x)) return false;
    if (!extract_named_double(s, "\"red_roi\"", "\"y\"", tmp.red_roi.y)) return false;
    if (!extract_named_double(s, "\"red_roi\"", "\"w\"", tmp.red_roi.w)) return false;
    if (!extract_named_double(s, "\"red_roi\"", "\"h\"", tmp.red_roi.h)) return false;
    if (!extract_named_double(s, "\"image_roi\"", "\"x\"", tmp.image_roi.x)) return false;
    if (!extract_named_double(s, "\"image_roi\"", "\"y\"", tmp.image_roi.y)) return false;
    if (!extract_named_double(s, "\"image_roi\"", "\"w\"", tmp.image_roi.w)) return false;
    if (!extract_named_double(s, "\"image_roi\"", "\"h\"", tmp.image_roi.h)) return false;

    if (!clamp_and_validate_roi(tmp.red_roi)) return false;
    if (!clamp_and_validate_roi(tmp.image_roi)) return false;
    rois = tmp;
    return true;
}

} // namespace vision_app
