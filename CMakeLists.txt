cmake_minimum_required(VERSION 3.8)
project(ShapeMatch)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})

include(FindOpenCV)

include_directories(${OpenCV4_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src)

#add_library(ShapeMatch src/ShapeMatch.cpp src/ShapeMatch.h)
#target_link_libraries(ShapeMatch debug ${OpenCV4_DEBUG_LIB} optimized ${OpenCV4_RELEASE_LIB})

add_library(ShapeMatchV2 src/ShapeMatchV2.cpp src/ShapeMatchV2.h)
target_link_libraries(ShapeMatchV2 debug ${OpenCV4_DEBUG_LIB} optimized ${OpenCV4_RELEASE_LIB})

#add_executable(MainTest example/main.cpp)
#target_link_libraries(MainTest ShapeMatch)

add_executable(MainTestV2 example/mainV2.cpp)
target_link_libraries(MainTestV2 ShapeMatchV2)


