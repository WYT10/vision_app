#include "modes.hpp"
#include <iostream>

namespace vision_app {

bool run_probe(const AppConfig& cfg, std::string& err) {
    std::cout << "[probe] task=" << cfg.probe.task << "\n";
    std::cout << "Implement: list/live/snap/bench through capture layer.\n";
    err.clear();
    return true;
}

bool run_calibrate(const AppConfig& cfg, std::string& err) {
    std::cout << "[calibrate]\n";
    std::cout << "ROI mode=" << (cfg.roi_mode == RoiMode::Fixed ? "fixed" : "dynamic_red_stacked") << "\n";
    std::cout << "Implement search->lock->tune->save loop here.\n";
    err.clear();
    return true;
}

bool run_deploy(const AppConfig& cfg, std::string& err) {
    std::cout << "[deploy]\n";
    std::cout << "Implement warp->trigger->roi->optional model loop here.\n";
    err.clear();
    return true;
}

} // namespace vision_app
