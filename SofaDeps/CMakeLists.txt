cmake_minimum_required(VERSION 2.8.12)

set(SOLUTION_NAME "SofaDeps")
project(${SOLUTION_NAME})

# Temporary variable until we move the sub projects
set( SofaDeps_ACTUAL_SOURCE_DIR ${SofaDeps_SOURCE_DIR}/../extlibs)

# extlibs
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/tinyxml ${CMAKE_CURRENT_BINARY_DIR}/tinyxml)

set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib.")
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/gtest ${CMAKE_CURRENT_BINARY_DIR}/gtest)
