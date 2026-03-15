#include "calibration.h"
#include "camera.h"
#include "config.h"
#include "deploy.h"
#include <iostream>

using namespace app;

static std::string argValue(int argc, char** argv, const std::string& key, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) return argv[i + 1];
    }
    return fallback;
}

int main(int argc, char** argv) {
    const std::string mode = (argc > 1) ? argv[1] : "probe";
    const std::string config_path = argValue(argc, argv, std::string("--config"), std::string("config/system_config.json"));

    if (mode == "new-profile") {
        AppConfig cfg = ProfileStore::makeDefault();
        std::string err;
        if (!ProfileStore::save(config_path, cfg, &err)) {
            std::cerr << "failed to create profile: " << err << std::endl;
            return 1;
        }
        std::cout << "created profile: " << config_path << std::endl;
        return 0;
    }

    AppConfig cfg;
    std::string err;
    if (!ProfileStore::load(config_path, cfg, &err)) {
        std::cerr << "failed to load config: " << err << std::endl;
        return 1;
    }

    if (mode == "probe") {
        std::vector<ProbeRow> rows;
        std::string e1, e2, e3;
        ProbeRunner::writeV4L2Report(cfg, &e1);
        ProbeRunner::runOpenCvProbe(cfg, rows, &e2);
        ProbeRunner::writeCsv(cfg.probe.csv_path, rows, &e3);
        ProbeRunner::writeJson(cfg.probe.json_path, cfg, rows, &e3);
        if (!e1.empty()) std::cerr << "probe v4l2 warning: " << e1 << std::endl;
        if (!e2.empty()) std::cerr << "probe OpenCV warning: " << e2 << std::endl;
        if (!e3.empty()) std::cerr << "probe write warning: " << e3 << std::endl;
        std::cout << "probe finished" << std::endl;
        return 0;
    }

    if (mode == "calibrate") {
        return runCalibration(cfg, config_path);
    }

    if (mode == "deploy") {
        return runDeploy(cfg);
    }

    std::cerr << "unknown mode: " << mode << std::endl;
    return 1;
}
