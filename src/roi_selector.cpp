#include "roi_selector.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "warp.hpp"

namespace app {

static void onMouse(int event, int x, int y, int, void* userdata) {
    auto* state = static_cast<SelectionState*>(userdata);
    if (!state || !state->active) return;

    if (event == cv::EVENT_LBUTTONDOWN) {
        state->dragging = true;
        state->start = cv::Point(x, y);
        state->current = state->start;
    } else if (event == cv::EVENT_MOUSEMOVE && state->dragging) {
        state->current = cv::Point(x, y);
    } else if (event == cv::EVENT_LBUTTONUP && state->dragging) {
        state->dragging = false;
        state->current = cv::Point(x, y);

        const int x0 = std::min(state->start.x, state->current.x);
        const int y0 = std::min(state->start.y, state->current.y);
        const int x1 = std::max(state->start.x, state->current.x);
        const int y1 = std::max(state->start.y, state->current.y);
        cv::Rect rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));

        if (state->target) {
            *state->target = normalizeRect(rect, state->ref_w, state->ref_h);
        }
        state->finished = true;
        state->active = false;
    }
}

void beginSelection(SelectionState& state, RoiNorm* target, int ref_w, int ref_h, const std::string& label) {
    state.active = true;
    state.finished = false;
    state.dragging = false;
    state.target = target;
    state.ref_w = std::max(ref_w, 1);
    state.ref_h = std::max(ref_h, 1);
    state.label = label;
}

void attachSelectionMouseCallback(const std::string& window_name, SelectionState* state) {
    cv::setMouseCallback(window_name, onMouse, state);
}

void drawSelectionOverlay(cv::Mat& image, const SelectionState& state) {
    if (state.target) {
        const cv::Rect rect = denormalizeRoi(*state.target, image.cols, image.rows);
        cv::rectangle(image, rect, cv::Scalar(255, 255, 0), 2);
        cv::putText(image, state.label, rect.tl() + cv::Point(4, 18), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);
    }

    if (state.active && state.dragging) {
        const cv::Rect rect(
            std::min(state.start.x, state.current.x),
            std::min(state.start.y, state.current.y),
            std::max(1, std::abs(state.current.x - state.start.x)),
            std::max(1, std::abs(state.current.y - state.start.y))
        );
        cv::rectangle(image, rect, cv::Scalar(0, 255, 255), 2);
    }
}

} // namespace app
