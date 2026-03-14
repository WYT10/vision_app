cmake_minimum_required(VERSION 3.10)
project(vision_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio objdetect calib3d)

include_directories(include ${OpenCV_INCLUDE_DIRS})

set(SOURCES
    src/app_main.cpp
    src/app_config.cpp
    src/app_camera.cpp
    src/app_calib.cpp
)

add_executable(vision_app ${SOURCES})
target_link_libraries(vision_app ${OpenCV_LIBS})