cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Compat LANGUAGES CXX)

set(SOFACOMPONENTCOMPATSRC_ROOT "src")

include(Sofa.Component.ODESolver.Backward.cmake)
include(Sofa.Component.ODESolver.Forward.cmake)
include(Sofa.Component.IO.Mesh.cmake)
include(Sofa.Component.Playback.cmake)

set(SOURCE_FILES
    ${SOFACOMPONENTCOMPATSRC_ROOT}/init.cpp
)

sofa_find_package(Sofa.Config REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Config)
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    OPTIMIZE_BUILD_DIR 0
)
