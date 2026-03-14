cmake_minimum_required(VERSION 3.16)
project(camera_combo_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio imgcodecs aruco)
find_package(nlohmann_json REQUIRED)

add_executable(camera_combo_app
    src/main.cpp
    src/app.cpp
    src/utils.cpp
    src/config.cpp
    src/cli.cpp
    src/camera.cpp
    src/probe.cpp
    src/tag_detector.cpp
    src/warp.cpp
    src/roi_selector.cpp
    src/calibration.cpp
    src/deploy.cpp
)

target_include_directories(camera_combo_app PRIVATE inc)
target_link_libraries(camera_combo_app PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json)

if(MSVC)
    target_compile_options(camera_combo_app PRIVATE /W4)
else()
    target_compile_options(camera_combo_app PRIVATE -Wall -Wextra -Wpedantic)
endif()
