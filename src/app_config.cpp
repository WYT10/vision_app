#include "app_config.hpp"

#include <cctype>
#include <fstream>
#include <iostream>
#include <string>

namespace vision_app {

static std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a])) != 0) ++a;
    size_t b = s.size();
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])) != 0) --b;
    return s.substr(a, b - a);
}

static bool parse_bool(const std::string& v) {
    return v == "1" || v == "true" || v == "TRUE" || v == "yes" || v == "on";
}

static IoMode parse_io_mode(const std::string& s) {
    if (s == "read") return IoMode::Read;
    return IoMode::Grab;
}

bool load_config_file(const std::string& path, AppOptions& opt) {
    std::ifstream in(path);
    if (!in.is_open()) return false;

    std::string line;
    while (std::getline(in, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        const size_t pos = line.find('=');
        if (pos == std::string::npos) continue;

        const std::string key = trim(line.substr(0, pos));
        const std::string val = trim(line.substr(pos + 1));

        if (key == "device") opt.device = val;
        else if (key == "width") opt.width = std::stoi(val);
        else if (key == "height") opt.height = std::stoi(val);
        else if (key == "fps") opt.fps = std::stoi(val);
        else if (key == "fourcc") opt.fourcc = val;
        else if (key == "duration_sec") opt.duration_sec = std::stoi(val);
        else if (key == "warmup_frames") opt.warmup_frames = std::stoi(val);
        else if (key == "probe_only") opt.probe_only = parse_bool(val);
        else if (key == "list_only") opt.list_only = parse_bool(val);
        else if (key == "headless") opt.headless = parse_bool(val);
        else if (key == "show_preview") opt.show_preview = parse_bool(val);
        else if (key == "save_csv") opt.save_csv = parse_bool(val);
        else if (key == "save_md") opt.save_md = parse_bool(val);
        else if (key == "report_dir") opt.report_dir = val;
        else if (key == "csv_path") opt.csv_path = val;
        else if (key == "probe_csv_path") opt.probe_csv_path = val;
        else if (key == "markdown_path") opt.markdown_path = val;
        else if (key == "capture_api") opt.capture_api = val;
        else if (key == "io_mode") opt.io_mode = parse_io_mode(val);
        else if (key == "buffer_size") opt.buffer_size = std::stoi(val);
        else if (key == "latest_only") opt.latest_only = parse_bool(val);
        else if (key == "drain_grabs") opt.drain_grabs = std::stoi(val);
    }
    return true;
}

void print_help() {
    std::cout
        << "vision_app options\n"
        << "  --device /dev/video0\n"
        << "  --width 1280\n"
        << "  --height 720\n"
        << "  --fps 30\n"
        << "  --fourcc MJPG\n"
        << "  --duration 10\n"
        << "  --warmup 8\n"
        << "  --probe-only\n"
        << "  --headless\n"
        << "  --preview\n"
        << "  --save-csv\n"
        << "  --save-md\n"
        << "  --report-dir ../report\n"
        << "  --csv-path ../report/test_results.csv\n"
        << "  --probe-csv-path ../report/probe_table.csv\n"
        << "  --markdown-path ../report/latest_report.md\n"
        << "  --capture-api v4l2|any\n"
        << "  --io-mode read|grab\n"
        << "  --buffer-size 1\n"
        << "  --latest-only\n"
        << "  --drain-grabs 3\n"
        << "  --config ../vision_app.conf\n"
        << "  -h --help\n";
}

bool parse_args(int argc, char** argv, AppOptions& opt, std::string& err) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];

        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                err = std::string("missing value for ") + name;
                return {};
            }
            return argv[++i];
        };

        if (a == "-h" || a == "--help") {
            print_help();
            err.clear();
            return false;
        } else if (a == "--device") opt.device = need("--device");
        else if (a == "--width") opt.width = std::stoi(need("--width"));
        else if (a == "--height") opt.height = std::stoi(need("--height"));
        else if (a == "--fps") opt.fps = std::stoi(need("--fps"));
        else if (a == "--fourcc") opt.fourcc = need("--fourcc");
        else if (a == "--duration") opt.duration_sec = std::stoi(need("--duration"));
        else if (a == "--warmup") opt.warmup_frames = std::stoi(need("--warmup"));
        else if (a == "--probe-only") opt.probe_only = true;
        else if (a == "--list-only") opt.list_only = true;
        else if (a == "--headless") { opt.headless = true; opt.show_preview = false; }
        else if (a == "--preview") { opt.headless = false; opt.show_preview = true; }
        else if (a == "--save-csv") opt.save_csv = true;
        else if (a == "--save-md") opt.save_md = true;
        else if (a == "--report-dir") opt.report_dir = need("--report-dir");
        else if (a == "--csv-path") opt.csv_path = need("--csv-path");
        else if (a == "--probe-csv-path") opt.probe_csv_path = need("--probe-csv-path");
        else if (a == "--markdown-path") opt.markdown_path = need("--markdown-path");
        else if (a == "--capture-api") opt.capture_api = need("--capture-api");
        else if (a == "--io-mode") opt.io_mode = parse_io_mode(need("--io-mode"));
        else if (a == "--buffer-size") opt.buffer_size = std::stoi(need("--buffer-size"));
        else if (a == "--latest-only") opt.latest_only = true;
        else if (a == "--drain-grabs") opt.drain_grabs = std::stoi(need("--drain-grabs"));
        else if (a == "--config") {
            opt.config_path = need("--config");
            load_config_file(opt.config_path, opt);
        } else {
            err = "unknown argument: " + a;
            return false;
        }

        if (!err.empty()) return false;
    }
    return true;
}

} // namespace vision_app
