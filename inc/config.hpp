#pragma once

#include "types.hpp"

namespace app {

AppConfig defaultConfig();
AppConfig loadConfig(const fs::path& path);
void saveConfig(const AppConfig& config, const fs::path& path);

} // namespace app
