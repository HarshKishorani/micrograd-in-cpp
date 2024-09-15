cmake_minimum_required(VERSION 3.20)

project(micrograd-in-cpp)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 20)

# Add the "inc" directory to the include search path
include_directories(${CMAKE_SOURCE_DIR}/inc)

# Find all source files in the "src" folder
file(GLOB_RECURSE SOURCE_FILES src/*.cpp src/*.c)

# Add the executable
add_executable(engine ${SOURCE_FILES})
