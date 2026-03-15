#include "calibration.h"
#include "camera.h"
#include "config.h"
#include "deploy.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
struct CliOptions
{
    std::string mode = "probe";
    std::string config_path = "config/system_config.sample.json";
    bool force_ui = false;
    bool force_no_ui = false;
    int camera_index = -1;
    int width = -1;
    int height = -1;
    int fps = -1;
    std::string fourcc;
    std::string backend;
};

void print_help()
{
    std::cout
        << "Usage: vision_app <probe|calibrate|deploy> [options]\n"
        << "Options:\n"
        << "  --config PATH       Config file path\n"
        << "  --camera N          Override camera index\n"
        << "  --width N           Override width\n"
        << "  --height N          Override height\n"
        << "  --fps N             Override fps\n"
        << "  --fourcc STR        Override FOURCC, e.g. MJPG\n"
        << "  --backend NAME      Override backend, e.g. V4L2\n"
        << "  --show-ui           Force UI on\n"
        << "  --no-ui             Force UI off\n";
}

bool parse_cli(int argc, char** argv, CliOptions& opts, std::string& error)
{
    if (argc >= 2)
        opts.mode = argv[1];

    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& flag) -> std::string {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + flag);
            return argv[++i];
        };

        try
        {
            if (arg == "--config")
                opts.config_path = require_value(arg);
            else if (arg == "--camera")
                opts.camera_index = std::stoi(require_value(arg));
            else if (arg == "--width")
                opts.width = std::stoi(require_value(arg));
            else if (arg == "--height")
                opts.height = std::stoi(require_value(arg));
            else if (arg == "--fps")
                opts.fps = std::stoi(require_value(arg));
            else if (arg == "--fourcc")
                opts.fourcc = require_value(arg);
            else if (arg == "--backend")
                opts.backend = require_value(arg);
            else if (arg == "--show-ui")
                opts.force_ui = true;
            else if (arg == "--no-ui")
                opts.force_no_ui = true;
            else if (arg == "--help" || arg == "-h")
            {
                print_help();
                std::exit(0);
            }
            else
            {
                error = "Unknown argument: " + arg;
                return false;
            }
        }
        catch (const std::exception& e)
        {
            error = e.what();
            return false;
        }
    }

    return true;
}

void apply_overrides(const CliOptions& cli, AppConfig& cfg)
{
    if (cli.camera_index >= 0)
        cfg.camera.device_index = cli.camera_index;
    if (cli.width > 0)
        cfg.camera.requested_mode.width = cli.width;
    if (cli.height > 0)
        cfg.camera.requested_mode.height = cli.height;
    if (cli.fps > 0)
        cfg.camera.requested_mode.fps = cli.fps;
    if (!cli.fourcc.empty())
        cfg.camera.requested_mode.fourcc = normalize_fourcc_string(cli.fourcc);
    if (!cli.backend.empty())
        cfg.camera.backend = backend_from_string(cli.backend);
    if (cli.force_ui)
    {
        cfg.runtime.show_ui = true;
        cfg.runtime.headless_deploy = false;
    }
    if (cli.force_no_ui)
    {
        cfg.runtime.show_ui = false;
        cfg.runtime.headless_deploy = true;
    }
}
}

int main(int argc, char** argv)
{
    CliOptions cli;
    std::string cli_error;
    if (!parse_cli(argc, argv, cli, cli_error))
    {
        std::cerr << cli_error << '\n';
        print_help();
        return 1;
    }

    AppConfig cfg;
    std::string cfg_error;
    if (!load_config(cli.config_path, cfg, &cfg_error))
    {
        std::cerr << "Config load failed: " << cfg_error << '\n';
        return 1;
    }

    apply_overrides(cli, cfg);

    if (cli.mode == "probe")
    {
        const auto results = run_camera_probe(cfg);
        std::string json_path;
        std::string csv_path;
        if (!write_probe_report(cfg.runtime.report_dir, results, &json_path, &csv_path))
        {
            std::cerr << "Failed to write probe report\n";
            return 1;
        }
        std::cout << "Probe report written:\n  JSON: " << json_path << "\n  CSV:  " << csv_path << '\n';
        return 0;
    }

    if (cli.mode == "calibrate")
    {
        std::string error;
        if (!run_calibration(cfg, cli.config_path, &error))
        {
            std::cerr << "Calibration failed: " << error << '\n';
            return 1;
        }
        return 0;
    }

    if (cli.mode == "deploy")
    {
        std::string error;
        if (!run_deploy(cfg, &error))
        {
            std::cerr << "Deploy failed: " << error << '\n';
            return 1;
        }
        return 0;
    }

    std::cerr << "Unknown mode: " << cli.mode << '\n';
    print_help();
    return 1;
}
