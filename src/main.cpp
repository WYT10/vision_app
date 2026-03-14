#include "camera.h"
#include "calibration.h"
#include "config.h"
#include "deploy.h"

#include <iostream>
#include <string>
#include <stdexcept>

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
};

void print_help()
{
    std::cout
        << "Usage: camera_combo_vision_app <probe|calibrate|deploy> [options]\n"
        << "Options:\n"
        << "  --config PATH       Config file path\n"
        << "  --camera N          Override camera index\n"
        << "  --width N           Override width\n"
        << "  --height N          Override height\n"
        << "  --fps N             Override fps\n"
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
        cfg.camera.index = cli.camera_index;
    if (cli.width > 0)
        cfg.camera.width = cli.width;
    if (cli.height > 0)
        cfg.camera.height = cli.height;
    if (cli.fps > 0)
        cfg.camera.fps = cli.fps;
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
        std::cerr << cli_error << "\n";
        print_help();
        return 1;
    }

    AppConfig cfg;
    std::string cfg_error;
    if (!load_config(cli.config_path, cfg, &cfg_error))
    {
        std::cerr << "Config load failed: " << cfg_error << "\n";
        return 1;
    }

    apply_overrides(cli, cfg);

    if (cli.mode == "probe")
    {
        const auto results = run_camera_probe(cfg);
        std::string json_path, csv_path;
        if (!write_probe_report(cfg.probe.report_dir, results, &json_path, &csv_path))
        {
            std::cerr << "Failed to write probe report\n";
            return 1;
        }
        std::cout << "Probe report written:\n  JSON: " << json_path << "\n  CSV:  " << csv_path << "\n";
        return 0;
    }

    if (cli.mode == "calibrate")
    {
        std::string error;
        if (!run_calibration(cfg, cli.config_path, &error))
        {
            std::cerr << "Calibration failed: " << error << "\n";
            return 1;
        }
        std::cout << "Calibration finished. Save with [S] during UI, or inspect updated config.\n";
        return 0;
    }

    if (cli.mode == "deploy")
    {
        std::string error;
        if (!run_deploy(cfg, &error))
        {
            std::cerr << "Deploy failed: " << error << "\n";
            return 1;
        }
        return 0;
    }

    std::cerr << "Unknown mode: " << cli.mode << "\n";
    print_help();
    return 1;
}
