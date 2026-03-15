cmake_minimum_required(VERSION 3.16)
project(vision_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.10 REQUIRED COMPONENTS core imgproc highgui videoio)

add_executable(vision_app
    src/main.c
    src/app_config.cpp
    src/camera_probe.cpp
    src/camera_test.cpp
    src/stats.cpp
)

# Force main.c to compile as C++
set_source_files_properties(src/main.c PROPERTIES LANGUAGE CXX)

target_include_directories(vision_app PRIVATE inc)
target_link_libraries(vision_app PRIVATE ${OpenCV_LIBS})

target_compile_options(vision_app PRIVATE
    -O3
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vision_app PRIVATE -g)
endif()