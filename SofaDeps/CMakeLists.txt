cmake_minimum_required(VERSION 2.8.12)

project(SofaDeps)

# Temporary variable until we move the sub projects
set( SofaDeps_ACTUAL_SOURCE_DIR ${SofaDeps_SOURCE_DIR}/../extlibs)

# extlibs mandatory
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/miniBoost ${CMAKE_CURRENT_BINARY_DIR}/miniBoost)
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/tinyxml ${CMAKE_CURRENT_BINARY_DIR}/tinyxml)
set(EIGEN_ROOT ${SofaDeps_ACTUAL_SOURCE_DIR}/eigen-3.2.1 CACHE PATH "Directory for Eigen3 find_package")
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/eigen-3.2.1 ${CMAKE_CURRENT_BINARY_DIR}/eigen)
set(gtest_force_shared_crt ON CACHE BOOL "Use shared (DLL) run-time lib even when Google Test is built as static lib.")
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/gtest ${CMAKE_CURRENT_BINARY_DIR}/gtest)

if(MSVC)
  set_target_properties(tinyxml    PROPERTIES FOLDER "SofaExtlibs")
  set_target_properties(gtest      PROPERTIES FOLDER "SofaExtlibs")
  set_target_properties(gtest_main PROPERTIES FOLDER "SofaExtlibs")
endif()
  
# optional extlibs
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/miniFlowVR ${CMAKE_CURRENT_BINARY_DIR}/miniFlowVR)
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/newmat ${CMAKE_CURRENT_BINARY_DIR}/newmat)

if(MSVC)
  set_target_properties(miniFlowVR PROPERTIES FOLDER "SofaExtlibs")
  set_target_properties(newmat     PROPERTIES FOLDER "SofaExtlibs")
endif()