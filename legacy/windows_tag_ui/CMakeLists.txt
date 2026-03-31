cmake_minimum_required(VERSION 3.16)
project(camera_plot_tag_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs videoio aruco)
find_package(nlohmann_json REQUIRED)

add_executable(camera_plot_tag_cpp src/main.cpp)

target_include_directories(camera_plot_tag_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera_plot_tag_cpp PRIVATE ${OpenCV_LIBS} nlohmann_json::nlohmann_json)