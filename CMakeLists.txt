project(test LANGUAGES CUDA CXX)
set(CMAKE_CUDA_ARCHITECTURES "80")

add_library(test_moduleA SHARED test_moduleA.cu)
add_library(test_moduleB SHARED test_moduleB.cu)
target_link_libraries(test_moduleB test_moduleA)
add_executable(test_main test_main.cc)
target_link_libraries(test_main test_moduleB)
