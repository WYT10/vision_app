cmake_minimum_required(VERSION 3.16)
project(vision_app LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs videoio aruco)
find_package(nlohmann_json REQUIRED)

add_executable(vision_app
    src/main.cpp
    src/config.cpp
    src/camera.cpp
    src/calibration.cpp
    src/deploy.cpp
    src/homography.cpp
)

target_include_directories(vision_app PRIVATE
    inc
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(vision_app PRIVATE
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

if(MSVC)
    target_compile_options(vision_app PRIVATE /W4 /permissive-)
else()
    target_compile_options(vision_app PRIVATE -O3 -DNDEBUG -Wall -Wextra -Wpedantic)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_output)
if(ipo_supported)
    set_property(TARGET vision_app PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()
