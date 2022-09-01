cmake_minimum_required(VERSION 3.12)
project(Sofa.GL.Component.Rendering2D LANGUAGES CXX)

set(SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR "src/sofa/gl/component/rendering2d")

set(HEADER_FILES
    ${SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR}/config.h.in
    ${SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR}/init.h
    ${SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR}/OglColorMap.h
    ${SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR}/OglLabel.h
    ${SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR}/OglViewport.h
)

set(SOURCE_FILES
    ${SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR}/init.cpp
    ${SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR}/OglColorMap.cpp
    ${SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR}/OglLabel.cpp
    ${SOFAGLCOMPONENTRENDRING2D_SOURCE_DIR}/OglViewport.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)
sofa_find_package(Sofa.GL REQUIRED)
sofa_find_package(Sofa.Component.Visual REQUIRED)
sofa_find_package(Sofa.Component.Setting REQUIRED)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core Sofa.GL Sofa.Component.Visual Sofa.Component.Setting)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_GL_COMPONENT_RENDERING2D_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_GL_COMPONENT_RENDERING2D_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
