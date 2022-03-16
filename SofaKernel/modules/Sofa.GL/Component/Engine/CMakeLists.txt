cmake_minimum_required(VERSION 3.12)
project(Sofa.GL.Component.Engine LANGUAGES CXX)

set(SOFAGLCOMPONENTENGINE_SOURCE_DIR "src/sofa/gl/component/engine")

set(HEADER_FILES
    ${SOFAGLCOMPONENTENGINE_SOURCE_DIR}/config.h.in
    ${SOFAGLCOMPONENTENGINE_SOURCE_DIR}/init.h
    ${SOFAGLCOMPONENTENGINE_SOURCE_DIR}/TextureInterpolation.h
    ${SOFAGLCOMPONENTENGINE_SOURCE_DIR}/TextureInterpolation.inl
)

set(SOURCE_FILES
    ${SOFAGLCOMPONENTENGINE_SOURCE_DIR}/init.cpp
    ${SOFAGLCOMPONENTENGINE_SOURCE_DIR}/TextureInterpolation.cpp
)

sofa_find_package(Sofa.GL REQUIRED)
sofa_find_package(Sofa.SimulationCore REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.SimulationCore Sofa.GL)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
