cmake_minimum_required(VERSION 3.16)
project(portable_cls_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_ONNX_RUNTIME "Build ONNX Runtime backend" ON)
option(ENABLE_NCNN "Build NCNN backend" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(portable_cls_infer
    src/main.cpp
)

target_include_directories(portable_cls_infer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(portable_cls_infer PRIVATE ${OpenCV_LIBS})

if(ENABLE_ONNX_RUNTIME)
    find_package(ONNXRuntime REQUIRED)
    target_compile_definitions(portable_cls_infer PRIVATE HAVE_ONNX_RUNTIME=1)
    target_link_libraries(portable_cls_infer PRIVATE ONNXRuntime::ONNXRuntime)
endif()

if(ENABLE_NCNN)
    find_package(ncnn REQUIRED)
    target_compile_definitions(portable_cls_infer PRIVATE HAVE_NCNN=1)
    target_link_libraries(portable_cls_infer PRIVATE ncnn)
endif()

if(NOT ENABLE_ONNX_RUNTIME AND NOT ENABLE_NCNN)
    message(FATAL_ERROR "Enable at least one backend: ENABLE_ONNX_RUNTIME or ENABLE_NCNN")
endif()
