#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace vision_app {

struct AprilTagConfig {
    std::string family = "auto"; // auto | 16 | 25 | 36
    int target_id = 0;
    bool require_target_id = true;
    bool manual_lock_only = true;
    int lock_frames = 4;
};

struct AprilTagDetection {
    bool found = false;
    std::string family;
    int id = -1;
    std::array<cv::Point2f,4> corners{};
    cv::Point2f center{};
};

struct RoiRatio { double x=0.08, y=0.08, w=0.18, h=0.18; };
struct RoiConfig { RoiRatio red_roi; RoiRatio image_roi{0.32,0.10,0.50,0.55}; };

struct DynamicRedRoiConfig {
    int band_y0 = 120;
    int band_y1 = 180;
    int search_x0 = 0;
    int search_x1 = -1;      // <=0 => full width
    int roi_gap_above_band = 0; // distance between image-roi bottom and red-band top
    int roi_anchor_y = -1;   // legacy override; <0 => compute from band_y0 - gap - roi_height
    int roi_width = 96;
    int roi_height = 96;
    int min_area = 40;
    int max_area = 0;        // <=0 => unbounded
    int morph_k = 3;
    int miss_tolerance = 5;
    int fallback_center_x = -1; // <0 => search-band center
    double center_alpha = 0.70; // new-detection smoothing weight
    bool show_mask_window = false;
};

struct WarpPackage {
    bool valid = false;
    cv::Mat H;          // 3x3 CV_64F, source -> warp
    cv::Mat Hinv;       // 3x3 CV_64F, warp -> source
    cv::Mat map1;       // remap maps
    cv::Mat map2;
    cv::Mat valid_mask; // CV_8UC1
    cv::Size src_size;
    cv::Size warp_size;
    std::string family;
    int id = -1;
    int target_tag_px = 0;
};

inline bool finite_pt(const cv::Point2f& p) {
    return std::isfinite(p.x) && std::isfinite(p.y);
}

inline double quad_area4(const std::array<cv::Point2f,4>& c) {
    double a = 0.0;
    for (int i = 0; i < 4; ++i) {
        const auto& p = c[i];
        const auto& q = c[(i + 1) % 4];
        a += static_cast<double>(p.x) * q.y - static_cast<double>(q.x) * p.y;
    }
    return 0.5 * std::abs(a);
}

inline std::vector<int> families_from_mode(const std::string& mode) {
    if (mode == "16") return {cv::aruco::DICT_APRILTAG_16h5};
    if (mode == "25") return {cv::aruco::DICT_APRILTAG_25h9};
    if (mode == "36") return {cv::aruco::DICT_APRILTAG_36h11};
    return {cv::aruco::DICT_APRILTAG_36h11, cv::aruco::DICT_APRILTAG_25h9, cv::aruco::DICT_APRILTAG_16h5};
}

inline std::string family_name_from_dict(int dict) {
    switch (dict) {
        case cv::aruco::DICT_APRILTAG_16h5: return "16";
        case cv::aruco::DICT_APRILTAG_25h9: return "25";
        default: return "36";
    }
}

inline bool detect_apriltag_best(const cv::Mat& frame,
                                 const AprilTagConfig& cfg,
                                 AprilTagDetection& det,
                                 std::string& err) {
    det = {};
    if (frame.empty()) { err = "empty frame"; return false; }
    cv::Mat gray;
    if (frame.channels() == 1) gray = frame;
    else cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    cv::aruco::DetectorParameters params;
    params.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    params.minMarkerPerimeterRate = 0.03;
    params.maxMarkerPerimeterRate = 4.0;

    for (const int dict_id : families_from_mode(cfg.family)) {
        auto dict = cv::aruco::getPredefinedDictionary(dict_id);
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<int> ids;
        cv::aruco::ArucoDetector detector(dict, params);
        detector.detectMarkers(gray, corners, ids);
        if (ids.empty()) continue;
        int best = 0;
        if (cfg.require_target_id) {
            best = -1;
            for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
                if (ids[i] == cfg.target_id) { best = i; break; }
            }
            if (best < 0) continue;
        }
        det.found = true;
        det.family = family_name_from_dict(dict_id);
        det.id = ids[best];
        for (int k = 0; k < 4; ++k) det.corners[k] = corners[best][k];
        det.center = 0.25f * (det.corners[0] + det.corners[1] + det.corners[2] + det.corners[3]);
        err.clear();
        return true;
    }
    err.clear();
    return true;
}

class TagLocker {
public:
    explicit TagLocker(int required_frames) : required_frames_(std::max(1, required_frames)) {}
    void reset() { hist_.clear(); }
    bool update(const AprilTagDetection& d) {
        if (!d.found) { reset(); return false; }
        if (!hist_.empty() && (hist_.back().id != d.id || hist_.back().family != d.family)) reset();
        hist_.push_back(d);
        while (static_cast<int>(hist_.size()) > required_frames_) hist_.pop_front();
        return static_cast<int>(hist_.size()) >= required_frames_;
    }
private:
    int required_frames_;
    std::deque<AprilTagDetection> hist_;
};

inline RoiRatio clamp_roi(RoiRatio r) {
    auto clamp01 = [](double v){ return std::max(0.0, std::min(1.0, v)); };
    r.x = clamp01(r.x); r.y = clamp01(r.y); r.w = clamp01(r.w); r.h = clamp01(r.h);
    if (r.x + r.w > 1.0) r.w = std::max(0.01, 1.0 - r.x);
    if (r.y + r.h > 1.0) r.h = std::max(0.01, 1.0 - r.y);
    r.w = std::max(0.01, r.w); r.h = std::max(0.01, r.h);
    return r;
}

inline cv::Rect clamp_rect_xywh(int x, int y, int w, int h, const cv::Size& sz) {
    const int use_w = std::clamp(w, 1, std::max(1, sz.width));
    const int use_h = std::clamp(h, 1, std::max(1, sz.height));
    const int max_x = std::max(0, sz.width - use_w);
    const int max_y = std::max(0, sz.height - use_h);
    const int use_x = std::clamp(x, 0, max_x);
    const int use_y = std::clamp(y, 0, max_y);
    return {use_x, use_y, use_w, use_h};
}

inline cv::Rect roi_to_rect(const RoiRatio& rr, const cv::Size& sz) {
    RoiRatio r = clamp_roi(rr);
    int x = std::clamp(static_cast<int>(std::round(r.x * sz.width)), 0, std::max(0, sz.width - 1));
    int y = std::clamp(static_cast<int>(std::round(r.y * sz.height)), 0, std::max(0, sz.height - 1));
    int w = std::clamp(static_cast<int>(std::round(r.w * sz.width)), 1, std::max(1, sz.width - x));
    int h = std::clamp(static_cast<int>(std::round(r.h * sz.height)), 1, std::max(1, sz.height - y));
    return {x, y, w, h};
}

inline std::string normalize_roi_mode(const std::string& s) {
    if (s == "dynamic" || s == "dynamic-red" || s == "dynamic-red-x") return "dynamic-red-x";
    return "fixed";
}

inline bool is_dynamic_roi_mode(const std::string& s) {
    return normalize_roi_mode(s) == "dynamic-red-x";
}

inline cv::Rect dynamic_search_rect(const DynamicRedRoiConfig& cfg, const cv::Size& sz) {
    const int x0 = std::clamp(cfg.search_x0, 0, std::max(0, sz.width - 1));
    const int x1_raw = (cfg.search_x1 <= 0) ? sz.width : cfg.search_x1;
    const int x1 = std::clamp(x1_raw, x0 + 1, std::max(x0 + 1, sz.width));
    const int y0 = std::clamp(cfg.band_y0, 0, std::max(0, sz.height - 1));
    const int y1_raw = (cfg.band_y1 <= 0) ? sz.height : cfg.band_y1;
    const int y1 = std::clamp(y1_raw, y0 + 1, std::max(y0 + 1, sz.height));
    return {x0, y0, x1 - x0, y1 - y0};
}

inline int dynamic_image_roi_top_y(const DynamicRedRoiConfig& cfg) {
    if (cfg.roi_anchor_y >= 0) return cfg.roi_anchor_y;
    return cfg.band_y0 - std::max(0, cfg.roi_gap_above_band) - std::max(1, cfg.roi_height);
}

inline cv::Rect dynamic_image_roi_rect(int center_x, const DynamicRedRoiConfig& cfg, const cv::Size& sz) {
    const int roi_w = std::max(1, cfg.roi_width);
    const int roi_h = std::max(1, cfg.roi_height);
    const int top_y = dynamic_image_roi_top_y(cfg);
    const int x = center_x - roi_w / 2;
    return clamp_rect_xywh(x, top_y, roi_w, roi_h, sz);
}

inline void clamp_dynamic_red_cfg(DynamicRedRoiConfig& cfg, const cv::Size& sz) {
    cfg.band_y0 = std::clamp(cfg.band_y0, 0, std::max(0, sz.height - 1));
    cfg.band_y1 = std::clamp(cfg.band_y1, cfg.band_y0 + 1, std::max(cfg.band_y0 + 1, sz.height));
    cfg.search_x0 = std::clamp(cfg.search_x0, 0, std::max(0, sz.width - 1));
    if (cfg.search_x1 > 0) cfg.search_x1 = std::clamp(cfg.search_x1, cfg.search_x0 + 1, std::max(cfg.search_x0 + 1, sz.width));
    cfg.roi_gap_above_band = std::max(0, cfg.roi_gap_above_band);
    cfg.roi_width = std::clamp(cfg.roi_width, 1, std::max(1, sz.width));
    cfg.roi_height = std::clamp(cfg.roi_height, 1, std::max(1, sz.height));
    if (cfg.roi_anchor_y >= 0) cfg.roi_anchor_y = std::clamp(cfg.roi_anchor_y, 0, std::max(0, sz.height - cfg.roi_height));
    cfg.min_area = std::max(1, cfg.min_area);
    if (cfg.max_area > 0) cfg.max_area = std::max(cfg.min_area, cfg.max_area);
    cfg.morph_k = std::max(1, cfg.morph_k);
    if ((cfg.morph_k % 2) == 0) ++cfg.morph_k;
    cfg.miss_tolerance = std::max(0, cfg.miss_tolerance);
    cfg.center_alpha = std::clamp(cfg.center_alpha, 0.0, 1.0);
}

inline void tune_dynamic_red_cfg(DynamicRedRoiConfig& cfg, int key, int step_px, const cv::Size& sz) {
    const int s = std::max(1, step_px);
    switch (key) {
        case 'w': cfg.band_y0 -= s; cfg.band_y1 -= s; cfg.roi_anchor_y = -1; break;
        case 's': cfg.band_y0 += s; cfg.band_y1 += s; cfg.roi_anchor_y = -1; break;
        case 'i': cfg.band_y1 -= s; cfg.roi_anchor_y = -1; break;
        case 'k': cfg.band_y1 += s; cfg.roi_anchor_y = -1; break;
        case 'a': cfg.roi_width -= s; break;
        case 'd': cfg.roi_width += s; break;
        case 'z': cfg.roi_height -= s; cfg.roi_anchor_y = -1; break;
        case 'x': cfg.roi_height += s; cfg.roi_anchor_y = -1; break;
        case 'j': cfg.roi_gap_above_band = std::max(0, cfg.roi_gap_above_band - s); cfg.roi_anchor_y = -1; break;
        case 'l': cfg.roi_gap_above_band += s; cfg.roi_anchor_y = -1; break;
        default: return;
    }
    clamp_dynamic_red_cfg(cfg, sz);
}

inline bool save_rois_yaml(const std::string& path,
                           const RoiConfig& rois,
                           const DynamicRedRoiConfig* dyn = nullptr,
                           const std::string* roi_mode = nullptr) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    cv::FileStorage fs(path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) return false;
    fs << "red_roi" << "{" << "x" << rois.red_roi.x << "y" << rois.red_roi.y << "w" << rois.red_roi.w << "h" << rois.red_roi.h << "}";
    fs << "image_roi" << "{" << "x" << rois.image_roi.x << "y" << rois.image_roi.y << "w" << rois.image_roi.w << "h" << rois.image_roi.h << "}";
    if (dyn) {
        fs << "dynamic_red_roi" << "{";
        fs << "band_y0" << dyn->band_y0;
        fs << "band_y1" << dyn->band_y1;
        fs << "search_x0" << dyn->search_x0;
        fs << "search_x1" << dyn->search_x1;
        fs << "roi_gap_above_band" << dyn->roi_gap_above_band;
        fs << "roi_anchor_y" << dyn->roi_anchor_y;
        fs << "roi_width" << dyn->roi_width;
        fs << "roi_height" << dyn->roi_height;
        fs << "min_area" << dyn->min_area;
        fs << "max_area" << dyn->max_area;
        fs << "morph_k" << dyn->morph_k;
        fs << "miss_tolerance" << dyn->miss_tolerance;
        fs << "fallback_center_x" << dyn->fallback_center_x;
        fs << "center_alpha" << dyn->center_alpha;
        fs << "show_mask_window" << static_cast<int>(dyn->show_mask_window ? 1 : 0);
        fs << "}";
    }
    if (roi_mode) fs << "roi_mode" << normalize_roi_mode(*roi_mode);
    return true;
}

inline bool load_rois_yaml(const std::string& path,
                           RoiConfig& rois,
                           DynamicRedRoiConfig* dyn = nullptr,
                           std::string* roi_mode = nullptr) {
    if (!std::filesystem::exists(path)) return false;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    auto read_roi = [](const cv::FileNode& n, RoiRatio& r) {
        if (n.empty()) return false;
        n["x"] >> r.x; n["y"] >> r.y; n["w"] >> r.w; n["h"] >> r.h; r = clamp_roi(r); return true;
    };
    read_roi(fs["red_roi"], rois.red_roi);
    read_roi(fs["image_roi"], rois.image_roi);

    if (dyn) {
        const cv::FileNode n = fs["dynamic_red_roi"];
        if (!n.empty()) {
            n["band_y0"] >> dyn->band_y0;
            n["band_y1"] >> dyn->band_y1;
            n["search_x0"] >> dyn->search_x0;
            n["search_x1"] >> dyn->search_x1;
            if (!n["roi_gap_above_band"].empty()) n["roi_gap_above_band"] >> dyn->roi_gap_above_band;
            if (!n["roi_anchor_y"].empty()) n["roi_anchor_y"] >> dyn->roi_anchor_y;
            n["roi_width"] >> dyn->roi_width;
            n["roi_height"] >> dyn->roi_height;
            n["min_area"] >> dyn->min_area;
            n["max_area"] >> dyn->max_area;
            n["morph_k"] >> dyn->morph_k;
            n["miss_tolerance"] >> dyn->miss_tolerance;
            n["fallback_center_x"] >> dyn->fallback_center_x;
            n["center_alpha"] >> dyn->center_alpha;
            int show_mask_window = dyn->show_mask_window ? 1 : 0;
            if (!n["show_mask_window"].empty()) n["show_mask_window"] >> show_mask_window;
            dyn->show_mask_window = show_mask_window != 0;
            clamp_dynamic_red_cfg(*dyn, cv::Size(4096, 4096));
        }
    }

    if (roi_mode && !fs["roi_mode"].empty()) {
        fs["roi_mode"] >> *roi_mode;
        *roi_mode = normalize_roi_mode(*roi_mode);
    }
    return true;
}

inline bool build_centered_warp_package_from_detection_px(const AprilTagDetection& det,
                                                          const cv::Size& src_size,
                                                          int canvas_width,
                                                          int canvas_height,
                                                          int target_tag_px,
                                                          WarpPackage& pack,
                                                          std::string& err) {
    pack = {};
    if (!det.found) { err = "tag not found"; return false; }
    for (const auto& p : det.corners) {
        if (!finite_pt(p)) { err = "non-finite tag corner"; return false; }
    }
    if (quad_area4(det.corners) < 64.0) { err = "tag quadrilateral too small"; return false; }

    int W = std::clamp(canvas_width, 128, 2048);
    int H = std::clamp(canvas_height, 128, 2048);
    int L = std::clamp(target_tag_px, 16, std::min(W, H) - 2);
    const double cx = 0.5 * static_cast<double>(W);
    const double cy = 0.5 * static_cast<double>(H);
    const double half = 0.5 * static_cast<double>(L);

    std::vector<cv::Point2f> src_quad(4), dst_quad(4);
    for (int i = 0; i < 4; ++i) src_quad[i] = det.corners[i];
    dst_quad[0] = cv::Point2f(static_cast<float>(cx - half), static_cast<float>(cy - half));
    dst_quad[1] = cv::Point2f(static_cast<float>(cx + half), static_cast<float>(cy - half));
    dst_quad[2] = cv::Point2f(static_cast<float>(cx + half), static_cast<float>(cy + half));
    dst_quad[3] = cv::Point2f(static_cast<float>(cx - half), static_cast<float>(cy + half));

    cv::Mat Hm = cv::getPerspectiveTransform(src_quad, dst_quad);
    if (Hm.empty()) { err = "failed to compute homography"; return false; }
    cv::Mat Hinv = Hm.inv();
    if (Hinv.empty()) { err = "homography inversion failed"; return false; }

    cv::Mat mapx(H, W, CV_32FC1, cv::Scalar(-1.0f));
    cv::Mat mapy(H, W, CV_32FC1, cv::Scalar(-1.0f));
    cv::Mat valid(H, W, CV_8UC1, cv::Scalar(0));

    const cv::Matx33d Hinvx(
        Hinv.at<double>(0,0), Hinv.at<double>(0,1), Hinv.at<double>(0,2),
        Hinv.at<double>(1,0), Hinv.at<double>(1,1), Hinv.at<double>(1,2),
        Hinv.at<double>(2,0), Hinv.at<double>(2,1), Hinv.at<double>(2,2));

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const cv::Vec3d v(static_cast<double>(x), static_cast<double>(y), 1.0);
            const cv::Vec3d p = Hinvx * v;
            const double w = p[2];
            if (!std::isfinite(w) || std::abs(w) < 1e-12) continue;
            const double sx = p[0] / w;
            const double sy = p[1] / w;
            if (sx >= 0.0 && sy >= 0.0 && sx < src_size.width && sy < src_size.height) {
                mapx.at<float>(y, x) = static_cast<float>(sx);
                mapy.at<float>(y, x) = static_cast<float>(sy);
                valid.at<uchar>(y, x) = 255;
            }
        }
    }

    pack.valid = true;
    pack.H = Hm;
    pack.Hinv = Hinv;
    pack.map1 = mapx;
    pack.map2 = mapy;
    pack.valid_mask = valid;
    pack.src_size = src_size;
    pack.warp_size = {W, H};
    pack.family = det.family;
    pack.id = det.id;
    pack.target_tag_px = L;
    err.clear();
    return true;
}

inline bool build_warp_package_from_detection(const AprilTagDetection& det,
                                              const cv::Size& src_size,
                                              int warp_width,
                                              int warp_height,
                                              int target_tag_px,
                                              WarpPackage& pack,
                                              std::string& err) {
    return build_centered_warp_package_from_detection_px(det, src_size, warp_width, warp_height, target_tag_px, pack, err);
}

inline bool apply_warp(const cv::Mat& src, const WarpPackage& pack, cv::Mat& dst, cv::Mat* out_valid = nullptr) {
    if (!pack.valid || src.empty()) return false;
    cv::remap(src, dst, pack.map1, pack.map2, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(255,255,255));
    if (dst.empty()) return false;
    if (out_valid) *out_valid = pack.valid_mask;
    return true;
}

inline bool save_warp_package(const std::string& path, const WarpPackage& pack) {
    if (!pack.valid) return false;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
    cv::FileStorage fs(path, cv::FileStorage::WRITE | cv::FileStorage::FORMAT_YAML);
    if (!fs.isOpened()) return false;
    fs << "family" << pack.family;
    fs << "id" << pack.id;
    fs << "src_w" << pack.src_size.width << "src_h" << pack.src_size.height;
    fs << "warp_w" << pack.warp_size.width << "warp_h" << pack.warp_size.height;
    fs << "target_tag_px" << pack.target_tag_px;
    fs << "H" << pack.H;
    fs << "Hinv" << pack.Hinv;
    fs << "map1" << pack.map1;
    fs << "map2" << pack.map2;
    fs << "valid_mask" << pack.valid_mask;
    return true;
}

inline bool load_warp_package(const std::string& path, WarpPackage& pack) {
    if (!std::filesystem::exists(path)) return false;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) return false;
    pack = {};
    int sw=0, sh=0, ww=0, wh=0;
    fs["family"] >> pack.family;
    fs["id"] >> pack.id;
    fs["src_w"] >> sw; fs["src_h"] >> sh;
    fs["warp_w"] >> ww; fs["warp_h"] >> wh;
    if (!fs["target_tag_px"].empty()) fs["target_tag_px"] >> pack.target_tag_px;
    fs["H"] >> pack.H;
    fs["Hinv"] >> pack.Hinv;
    fs["map1"] >> pack.map1;
    fs["map2"] >> pack.map2;
    fs["valid_mask"] >> pack.valid_mask;
    pack.src_size = {sw, sh};
    pack.warp_size = {ww, wh};
    pack.valid = !pack.map1.empty() && !pack.valid_mask.empty();
    return pack.valid;
}

inline void draw_detection_overlay(cv::Mat& img, const AprilTagDetection& det) {
    if (!det.found) return;
    for (int i=0;i<4;++i) cv::line(img, det.corners[i], det.corners[(i+1)%4], cv::Scalar(0,255,255), 2);
    cv::circle(img, det.center, 4, cv::Scalar(255,0,255), -1);
    const std::string txt = "tag family=" + det.family + " id=" + std::to_string(det.id);
    cv::putText(img, txt, {12, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,255,255), 2);
}

inline void draw_rois(cv::Mat& img, const RoiConfig& rois, int selected) {
    const cv::Rect rr = roi_to_rect(rois.red_roi, img.size());
    const cv::Rect ir = roi_to_rect(rois.image_roi, img.size());
    cv::rectangle(img, rr, selected==0 ? cv::Scalar(0,0,255) : cv::Scalar(40,40,180), 2);
    cv::putText(img, "red_roi", rr.tl() + cv::Point(4,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
    cv::rectangle(img, ir, selected==1 ? cv::Scalar(255,0,0) : cv::Scalar(180,40,40), 2);
    cv::putText(img, "image_roi", ir.tl() + cv::Point(4,20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,0,0), 2);
}

inline void adjust_roi(RoiRatio& r, int key, double move_step, double size_step) {
    if (key=='w') r.y -= move_step;
    if (key=='s') r.y += move_step;
    if (key=='a') r.x -= move_step;
    if (key=='d') r.x += move_step;
    if (key=='j') r.w -= size_step;
    if (key=='l') r.w += size_step;
    if (key=='i') r.h -= size_step;
    if (key=='k') r.h += size_step;
    r = clamp_roi(r);
}

} // namespace vision_app
