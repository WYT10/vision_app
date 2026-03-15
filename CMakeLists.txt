cmake_minimum_required(VERSION 3.16)
project(vision_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(APP_ENABLE_UI "Enable OpenCV GUI windows" ON)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs videoio aruco calib3d)
find_package(nlohmann_json REQUIRED)

add_executable(vision_app
    src/main.cpp
    src/config.cpp
    src/camera.cpp
    src/homography.cpp
    src/calibration.cpp
    src/deploy.cpp
)

target_include_directories(vision_app PRIVATE inc)

target_link_libraries(vision_app PRIVATE
    opencv_core
    opencv_highgui
    opencv_imgproc
    opencv_imgcodecs
    opencv_videoio
    opencv_aruco
    opencv_calib3d
    nlohmann_json::nlohmann_json
)

if(APP_ENABLE_UI)
    target_compile_definitions(vision_app PRIVATE APP_ENABLE_UI=1)
else()
    target_compile_definitions(vision_app PRIVATE APP_ENABLE_UI=0)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(vision_app PRIVATE -O3 -DNDEBUG)
endif()
