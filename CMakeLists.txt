cmake_minimum_required(VERSION 4.0)
project(ModernRobotics)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR
        "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
set(warnings "-Wall -Wextra -Werror")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
 set(warnings "/W4 /WX /EHsc")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 11)



find_package (Eigen3 REQUIRED NO_MODULE)

add_library(ModernRobotics SHARED src/modern_robotics.cpp include/modern_robotics.h)
target_link_libraries(ModernRobotics Eigen3::Eigen)


install (TARGETS ModernRobotics
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
