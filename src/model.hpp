#pragma once

#include <string>
#include <vector>

namespace vision_app {

struct ModelConfig;
struct ModelResult;
struct RoiRuntimeData;

struct ModelHit {
    int index = -1;
    float score = 0.f;
    std::string label;
};

bool init_model_runtime(const ModelConfig& cfg, std::string& err);
void release_model_runtime();
bool model_runtime_ready();
std::string model_runtime_backend();

bool run_model_on_image_roi(const RoiRuntimeData& in,
                            const ModelConfig& cfg,
                            ModelResult& out,
                            std::string& err);

} // namespace vision_app
