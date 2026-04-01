cmake_minimum_required(VERSION 3.16)
project(vision_app_stacked_zones LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio aruco imgcodecs)

add_executable(vision_app
    src/main.cpp
    src/config.cpp
    src/modes.cpp
    src/red_trigger.cpp
    src/roi_runtime.cpp
    src/text_console.cpp
)

target_include_directories(vision_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(vision_app PRIVATE ${OpenCV_LIBS})
