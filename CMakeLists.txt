cmake_minimum_required(VERSION 3.16)
project(vision_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.10 REQUIRED COMPONENTS core imgproc highgui videoio aruco imgcodecs)
find_package(ncnn REQUIRED)
find_package(OpenMP)

add_executable(vision_app
    src/main.cpp
    src/app_config.cpp
    src/camera.cpp
    src/status_ui.cpp
    src/trigger.cpp
    src/model.cpp
)

target_include_directories(vision_app PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(vision_app PRIVATE ${OpenCV_LIBS} ncnn)

if (OpenMP_CXX_FOUND)
    target_link_libraries(vision_app PRIVATE OpenMP::OpenMP_CXX)
endif()

target_compile_options(vision_app PRIVATE -O3 -Wall -Wextra -Wpedantic)
