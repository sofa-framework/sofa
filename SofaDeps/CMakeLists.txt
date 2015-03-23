cmake_minimum_required(VERSION 2.8.12)

project(SofaDeps)

# Temporary variable until we move the sub projects
set(SofaDeps_ACTUAL_SOURCE_DIR ${SofaDeps_SOURCE_DIR}/../extlibs)

option(SOFA_USE_MINIBOOST
"Use the (header-only) subset of Boost distributed in the extlibs/ directory
(a.k.a. miniBoost).  This subset is sufficient to compile the framework and the
default components, but some plugins need other components of Boost." ON)
if(SOFA_USE_MINIBOOST)
    add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/miniBoost ${CMAKE_CURRENT_BINARY_DIR}/miniBoost)
endif()

add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/tinyxml ${CMAKE_CURRENT_BINARY_DIR}/tinyxml)
set(EIGEN_ROOT ${SofaDeps_ACTUAL_SOURCE_DIR}/eigen-3.2.1 CACHE PATH "Directory for Eigen3 find_package")
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/eigen-3.2.1 ${CMAKE_CURRENT_BINARY_DIR}/eigen)

set_target_properties(tinyxml PROPERTIES FOLDER "SofaExtlibs")

# optional extlibs
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/miniFlowVR ${CMAKE_CURRENT_BINARY_DIR}/miniFlowVR)
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/newmat ${CMAKE_CURRENT_BINARY_DIR}/newmat)
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/CImg ${CMAKE_CURRENT_BINARY_DIR}/CImg)
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/SuiteSparse ${CMAKE_CURRENT_BINARY_DIR}/SuiteSparse)
add_subdirectory(${SofaDeps_ACTUAL_SOURCE_DIR}/csparse ${CMAKE_CURRENT_BINARY_DIR}/csparse)

set_target_properties(miniFlowVR PROPERTIES FOLDER "SofaExtlibs")
set_target_properties(newmat PROPERTIES FOLDER "SofaExtlibs")
set_target_properties(csparse PROPERTIES FOLDER "SofaExtlibs")
