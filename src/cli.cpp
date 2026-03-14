#include "cli.hpp"

#include <iostream>
#include <stdexcept>

namespace app {

CliOptions parseArgs(int argc, char** argv) {
    CliOptions cli;
    if (argc >= 2) {
        cli.mode = argv[1];
    }

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        auto needValue = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for argument: " + arg);
            }
            return argv[++i];
        };

        if (arg == "--config") cli.config_path = needValue();
        else if (arg == "--camera") cli.camera_index = std::stoi(needValue());
        else if (arg == "--width") cli.width = std::stoi(needValue());
        else if (arg == "--height") cli.height = std::stoi(needValue());
        else if (arg == "--fps") cli.fps = std::stoi(needValue());
        else if (arg == "--tag-family") cli.tag_family = needValue();
        else if (arg == "--tag-id") cli.tag_id = std::stoi(needValue());
        else if (arg == "--flip") cli.flip = true;
        else if (arg == "--no-flip") cli.flip = false;
        else if (arg == "-h" || arg == "--help") cli.mode = "help";
        else throw std::runtime_error("Unknown argument: " + arg);
    }

    return cli;
}

void applyCliOverrides(const CliOptions& cli, AppConfig& config) {
    if (cli.camera_index) config.camera.index = *cli.camera_index;
    if (cli.width) config.camera.width = *cli.width;
    if (cli.height) config.camera.height = *cli.height;
    if (cli.fps) config.camera.fps = *cli.fps;
    if (cli.tag_family) {
        config.tag.mode = "family";
        config.tag.family = *cli.tag_family;
    }
    if (cli.tag_id) {
        config.tag.mode = "id";
        config.tag.id = *cli.tag_id;
    }
    if (cli.flip) config.camera.flip_horizontal = *cli.flip;
}

void printUsage() {
    std::cout
        << "Usage:\n"
        << "  camera_combo_app probe --config config/system_config.json\n"
        << "  camera_combo_app calibrate --config config/system_config.json\n"
        << "  camera_combo_app deploy --config config/system_config.json\n\n"
        << "Optional overrides:\n"
        << "  --camera N --width W --height H --fps FPS --tag-family NAME --tag-id ID --flip --no-flip\n";
}

} // namespace app
