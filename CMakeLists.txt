cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CXX_FLAGS "-Wall" "-pedantic")
set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS} -std=c++17 -O2")

project(camera_fusion)

find_package(LibXml2 REQUIRED)
find_package(OpenCV 4.1 REQUIRED)


include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})

# Executable for create matrix exercise
add_executable (2D_feature_tracking src/matching2D_Student.cpp src/MidTermProject_Camera_Student.cpp)
target_include_directories(2D_feature_tracking PRIVATE ${LIBXML2_INCLUDE_DIR})
target_link_libraries (2D_feature_tracking PRIVATE ${OpenCV_LIBRARIES}  ${LIBXML2_LIBRARIES})