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


install(TARGETS ${PROJECT_NAME}
        EXPORT MetisTargets
        RUNTIME DESTINATION bin COMPONENT Metis_libraries
        LIBRARY DESTINATION lib COMPONENT Metis_libraries
        ARCHIVE DESTINATION lib COMPONENT Metis_libraries
        PUBLIC_HEADER DESTINATION include COMPONENT Metis_headers)


include(CMakePackageConfigHelpers)

### MetisConfig.cmake
## Build tree
set(METIS_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
configure_package_config_file(MetisConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/MetisConfig.cmake
                              INSTALL_DESTINATION metis
                              PATH_VARS METIS_INCLUDE_DIR)
## Install tree
set(METIS_INCLUDE_DIR include)
configure_package_config_file(MetisConfig.cmake.in
                              ${CMAKE_CURRENT_BINARY_DIR}/InstalledMetisConfig.cmake
                              INSTALL_DESTINATION lib/metis/cmake
                              PATH_VARS METIS_INCLUDE_DIR)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/InstalledMetisConfig.cmake
        DESTINATION lib/metis/cmake
        RENAME MetisConfig.cmake)

## MetisTargets.cmake
install(EXPORT MetisTargets DESTINATION lib/metis/cmake)
