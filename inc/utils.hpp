#pragma once

#include <filesystem>
#include <string>

namespace app {
namespace fs = std::filesystem;

std::string nowStamp();
void ensureDir(const fs::path& path);
double clamp01(double x);
std::string lower(std::string s);

} // namespace app
