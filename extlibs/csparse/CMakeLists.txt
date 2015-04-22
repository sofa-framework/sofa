cmake_minimum_required(VERSION 2.8)
project(csparse)

set(HEADER_FILES
    csparse.h
    ldl.h
    UFconfig.h)

set(SOURCE_FILES
    csparse.c
    ldl.c)

# The code must be relocatable if we want to link a shared library against it.
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    add_definitions("-fPIC")
endif()

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER "SofaExtlibs")

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>")

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

install(TARGETS csparse
        EXPORT CSparseTargets
        RUNTIME DESTINATION bin COMPONENT CSparse_libraries
        LIBRARY DESTINATION lib COMPONENT CSparse_libraries
        ARCHIVE DESTINATION lib COMPONENT CSparse_libraries
        PUBLIC_HEADER DESTINATION include COMPONENT CSparse_headers)


include(CMakePackageConfigHelpers)

# CSparseConfig.cmake (build tree)
set(CSPARSE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
configure_package_config_file(CSparseConfig.cmake.in
                              ${CMAKE_BINARY_DIR}/CSparse/CSparseConfig.cmake
                              INSTALL_DESTINATION CSparse
                              PATH_VARS CSPARSE_INCLUDE_DIR)

# CSparseConfig.cmake (installed)
set(CSPARSE_INCLUDE_DIR include)
configure_package_config_file(CSparseConfig.cmake.in
                              ${CMAKE_BINARY_DIR}/CSparse/CSparseConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/CSparse
                              PATH_VARS CSPARSE_INCLUDE_DIR)

install(FILES ${CMAKE_BINARY_DIR}/CSparse/CSparseConfigInstall.cmake
        DESTINATION lib/cmake/CSparse
        RENAME CSparseConfig.cmake)

# CSparseTargets.cmake
install(EXPORT CSparseTargets
        DESTINATION lib/cmake/CSparse)
