cmake_minimum_required(VERSION 3.5)

project(infResnet LANGUAGES CXX)
set(CMAKE_PREFIX_PATH "libtorch/share/cmake/Torch")
set(Torch_DIR "/home/mihuzz/libtorch/share/cmake/Torch")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(TorchVision REQUIRED)
find_package(LAPACK REQUIRED)
#find_package(nlohmann_json 3.9.1 REQUIRED)
#find_package(Threads REQUIRED)

find_package(Python3 COMPONENTS Development)

add_executable(infResnet main.cpp)
target_link_libraries(infResnet ${OpenCV_LIBS})
target_link_libraries(infResnet ${TORCH_LIBRARIES} TorchVision::TorchVision Python3::Python)
target_link_libraries(infResnet TorchVision::TorchVision)
target_link_libraries(infResnet PRIVATE ${Python_LIBRARIES})
set_property(TARGET infResnet PROPERTY CXX_STANDARD 14)
