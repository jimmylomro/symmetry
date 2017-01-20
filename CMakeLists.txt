cmake_minimum_required(VERSION 2.6)

project(locky)

find_package(OpenCV REQUIRED)
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)

# locky
include_directories(include)
link_directories(${OpenCV_LIB_DIR})
add_library(locky_static STATIC  src/locky.cpp)
target_link_libraries(locky_static ${OpenV_LIBS})


# own usage example
link_directories(${OpenCV_LIB_DIR})
add_executable(locky_example src/locky_example.cpp)
target_link_libraries(locky_example locky_static ${OpenCV_LIBS})


