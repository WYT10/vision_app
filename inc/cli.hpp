#pragma once

#include "types.hpp"

namespace app {

CliOptions parseArgs(int argc, char** argv);
void applyCliOverrides(const CliOptions& cli, AppConfig& config);
void printUsage();

} // namespace app
