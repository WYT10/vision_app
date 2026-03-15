#include <iostream>
#include <string>

#include "vision_app.hpp"

int main(int argc, char** argv) {
    vision_app::AppOptions opt;
    std::string err;

    vision_app::load_config_file(opt.config_path, opt);

    if (!vision_app::parse_args(argc, argv, opt, err)) {
        if (!err.empty()) {
            std::cerr << "Argument error: " << err << "\n";
            return 1;
        }
        return 0;
    }

    vision_app::ProbeResult probe;
    if (!vision_app::probe_camera_modes(opt.device, probe, err)) {
        std::cerr << "Probe failed: " << err << "\n";
        return 1;
    }

    vision_app::print_probe_result(probe);

    if (!vision_app::ensure_report_dirs(opt, err)) {
        std::cerr << "Report directory error: " << err << "\n";
        return 1;
    }

    if (!vision_app::write_probe_csv(opt.probe_csv_path, probe)) {
        std::cerr << "Warning: failed to write probe CSV: " << opt.probe_csv_path << "\n";
    }

    if (opt.probe_only || opt.list_only) {
        return 0;
    }

    vision_app::RuntimeStats stats;
    if (!vision_app::run_camera_test(opt, stats, err)) {
        std::cerr << "Camera test failed: " << err << "\n";
        return 1;
    }

    vision_app::print_runtime_stats(opt, stats);

    if (opt.save_csv && !vision_app::append_test_csv(opt.csv_path, opt, stats)) {
        std::cerr << "Warning: failed to append CSV: " << opt.csv_path << "\n";
    }

    if (opt.save_md && !vision_app::write_markdown_report(opt.markdown_path, opt, probe, stats)) {
        std::cerr << "Warning: failed to write markdown report: " << opt.markdown_path << "\n";
    }

    return 0;
}
