cmake_minimum_required(VERSION 3.11)
include(ExternalProject)
include(FetchContent)
set(FETCHCONTENT_QUIET True)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(BUILD_TESTS OFF CACHE BOOL "xtl test suite")
set(WITH_LAPACK OFF CACHE BOOL "Include Lapack library support")

FetchContent_Declare(
  SDL
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG origin/main
)
FetchContent_MakeAvailable(SDL)

FetchContent_Declare(
  OpenCV
  GIT_REPOSITORY https://github.com/opencv/opencv.git
  GIT_TAG origin/4.x
)
FetchContent_MakeAvailable(OpenCV)

FetchContent_Declare(
  gl3w
  GIT_REPOSITORY https://github.com/bwrsandman/gl3w.git
)
FetchContent_MakeAvailable(gl3w)

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/pr0g/imgui.cmake.git
  GIT_TAG origin/main
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
  xtl
  GIT_REPOSITORY https://github.com/xtensor-stack/xtl.git
)
FetchContent_MakeAvailable(xtl)

FetchContent_Declare(
  xtensor
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor.git
)
FetchContent_MakeAvailable(xtensor)

FetchContent_Declare(
  xtensor-blas
  GIT_REPOSITORY https://github.com/xtensor-stack/xtensor-blas.git
)
FetchContent_MakeAvailable(xtensor-blas)

project(CPlusPlusNeuralNetwork)

add_subdirectory(NeuralNetwork)
add_subdirectory(TestApplication)

set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT TestApplication)