#include <iostream>
#include <string>
#include "params.hpp"
#include "config.hpp"
#include "modes.hpp"

int main(int argc, char** argv) {
    vision_app::AppConfig cfg;
    std::string err;

    if (!vision_app::load_default_config(cfg, err)) {
        std::cerr << "default config error: " << err << "\n";
        return 1;
    }
    if (!vision_app::load_config_from_argv(argc, argv, cfg, err)) {
        std::cerr << "config error: " << err << "\n";
        return 1;
    }

    std::cout << vision_app::dump_effective_config(cfg) << std::endl;

    if (cfg.mode == vision_app::AppMode::Probe) {
        return vision_app::run_probe(cfg, err) ? 0 : 1;
    }
    if (cfg.mode == vision_app::AppMode::Calibrate) {
        return vision_app::run_calibrate(cfg, err) ? 0 : 1;
    }
    if (cfg.mode == vision_app::AppMode::Deploy) {
        return vision_app::run_deploy(cfg, err) ? 0 : 1;
    }

    std::cerr << "unknown mode\n";
    return 1;
}
