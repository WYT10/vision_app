#include "app.hpp"

#include <iostream>
#include <stdexcept>

#include "calibration.hpp"
#include "cli.hpp"
#include "config.hpp"
#include "deploy.hpp"
#include "probe.hpp"

namespace app {

int runApp(int argc, char** argv) {
    try {
        const CliOptions cli = parseArgs(argc, argv);
        if (cli.mode == "help") {
            printUsage();
            return 0;
        }

        AppConfig config = loadConfig(cli.config_path);
        applyCliOverrides(cli, config);

        if (cli.mode == "probe") {
            const auto results = runProbe(config, "reports");
            std::cout << "Probe complete. Tested " << results.size() << " combinations. Reports written to ./reports\n";
            return 0;
        }
        if (cli.mode == "calibrate") {
            return runCalibration(config, cli.config_path);
        }
        if (cli.mode == "deploy") {
            return runDeploy(config);
        }

        printUsage();
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}

} // namespace app
