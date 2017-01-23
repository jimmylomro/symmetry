cmake_minimum_required(VERSION 2.6)

project(locky)

find_package(OpenCV REQUIRED)
MESSAGE(STATUS "LINK LIBRARIES ${OpenCV_LIBS}")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

IF(UNIX)
    IF(CMAKE_COMPILER_IS_GNUCXX)
         SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden")
    ENDIF(CMAKE_COMPILER_IS_GNUCXX)
ENDIF(UNIX)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/lib)


#------- Libraries

# own
include_directories(include/own)
link_directories(${OpenCV_LIB_DIR})
add_library(own_static STATIC  src/own/own.cpp src/own/nms.cpp)
target_link_libraries(own_static ${OpenV_LIBS})

# symbrisk
include_directories(include/symbrisk)
link_directories(${OpenCV_LIB_DIR})
add_library(symbrisk_static STATIC  src/symbrisk/symbrisk.cpp)
target_link_libraries(symbrisk_static ${OpenV_LIBS})

# dbscan
include_directories(include/dbscan)
link_directories(${OpenCV_LIB_DIR})
add_library(dbscan_static STATIC  src/dbscan/dbscan.cpp)
target_link_libraries(dbscan_static ${OpenV_LIBS})


#------- Examples

# own usage example
link_directories(${OpenCV_LIB_DIR})
add_executable(own_example src/own/own_example.cpp)
target_link_libraries(own_example own_static ${OpenCV_LIBS})

# symbrisk usage example
link_directories(${OpenCV_LIB_DIR})
add_executable(symbrisk_example src/symbrisk/symbrisk_example.cpp)
target_link_libraries(symbrisk_example symbrisk_static ${OpenCV_LIBS})
