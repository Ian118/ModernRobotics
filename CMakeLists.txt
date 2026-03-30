cmake_minimum_required(VERSION 3.10.0)
project(ModernRobotics)

option(BUILD_TESTING "Build the testing tree." OFF)

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(ModernRobotics INTERFACE)
target_include_directories(ModernRobotics INTERFACE include)
target_sources(ModernRobotics
    INTERFACE
    FILE_SET HEADERS
    FILES
    include/ModernRobotics
)
target_link_libraries(ModernRobotics INTERFACE Eigen3::Eigen)

if(BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(test_all test/test.cpp)
  target_link_libraries(test_all PRIVATE ModernRobotics GTest::gtest_main GTest::gtest Eigen3::Eigen)
  gtest_discover_tests(test_all)
endif()
