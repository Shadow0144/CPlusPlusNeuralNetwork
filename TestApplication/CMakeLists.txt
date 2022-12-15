cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(OpenCV REQUIRED)

file(GLOB_RECURSE HDR "hdr/*.h")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HDR})

file(GLOB_RECURSE SRC "src/*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SRC})

add_executable(TestApplication ${HDR} ${SRC})

target_include_directories(TestApplication PRIVATE
  "hdr")

target_link_libraries(TestApplication
  NeuralNetwork)