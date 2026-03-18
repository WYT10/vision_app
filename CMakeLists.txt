cmake_minimum_required(VERSION 3.16)
project(vision_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_ONNX_RUNTIME "Build ONNX Runtime backend" ON)
option(ENABLE_NCNN "Build NCNN backend" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCV 4.10 REQUIRED COMPONENTS core imgproc highgui videoio aruco imgcodecs)

add_executable(vision_app
    src/main.c
    src/roi_helper.cpp
    src/model.cpp
)
set_source_files_properties(src/main.c PROPERTIES LANGUAGE CXX)

target_include_directories(vision_app PRIVATE
    src
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vision_app PRIVATE ${OpenCV_LIBS})

if(ENABLE_ONNX_RUNTIME)
    find_package(ONNXRuntime REQUIRED)
    target_compile_definitions(vision_app PRIVATE HAVE_ONNX_RUNTIME=1)
    target_link_libraries(vision_app PRIVATE ONNXRuntime::ONNXRuntime)
endif()

if(ENABLE_NCNN)
    find_package(ncnn REQUIRED)
    target_compile_definitions(vision_app PRIVATE HAVE_NCNN=1)
    target_link_libraries(vision_app PRIVATE ncnn)
endif()

if(NOT ENABLE_ONNX_RUNTIME AND NOT ENABLE_NCNN)
    message(FATAL_ERROR "Enable at least one backend: ENABLE_ONNX_RUNTIME or ENABLE_NCNN")
endif()

target_compile_options(vision_app PRIVATE -O3 -Wall -Wextra -Wpedantic)
