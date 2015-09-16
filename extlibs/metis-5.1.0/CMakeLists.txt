cmake_minimum_required(VERSION 2.8.12)
project(metis)

unset(GKLIB_PATH CACHE)
get_filename_component(GKLIB_PATH "GKlib" ABSOLUTE)
include(GKlib/GKlibSystem.cmake)

file(GLOB metis_sources libmetis/*.c)

# The code must be relocatable if we want to link a shared library against it.
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    add_compile_options(-fPIC)
endif()

add_library(${PROJECT_NAME} STATIC ${metis_sources} ${GKlib_sources})
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/GKlib>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/libmetis>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER include/metis.h)
if(UNIX)
    target_link_libraries(${PROJECT_NAME} m)
endif()


include(../../SofaFramework/SofaPackageConfigHelpers.cmake)
sofa_create_package(Metis 5.1.0 metis "")
