cmake_minimum_required(VERSION 3.16)
project(vision_app LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4.10 REQUIRED)

add_executable(vision_app src/main.c)
set_source_files_properties(src/main.c PROPERTIES LANGUAGE CXX)

target_include_directories(vision_app PRIVATE src)
target_link_libraries(vision_app PRIVATE ${OpenCV_LIBS})

target_compile_options(vision_app PRIVATE -O3 -Wall -Wextra -Wpedantic)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(vision_app PRIVATE -g)
endif()
