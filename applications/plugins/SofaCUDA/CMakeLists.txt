cmake_minimum_required(VERSION 3.22)
project(SofaCUDA LANGUAGES CUDA CXX)

set(SOFACUDA_MAJOR_VERSION 0)
set(SOFACUDA_MINOR_VERSION 1)
set(SOFACUDA_VERSION ${SOFACUDA_MAJOR_VERSION}.${SOFACUDA_MINOR_VERSION})

find_package(Sofa.Config REQUIRED)

set(SOFACUDA_SOURCE_DIR "src/SofaCUDA")

sofa_add_subdirectory_modules(SOFACUDA_TARGETS
    DIRECTORIES
    Core
    Component
)

set(HEADER_FILES
    ${SOFACUDA_SOURCE_DIR}/config.h.in
    ${SOFACUDA_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACUDA_SOURCE_DIR}/init.cpp
)

set(TARGET_NAME ${PROJECT_NAME})

add_library(${TARGET_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${TARGET_NAME} PUBLIC Sofa.Config)
target_link_libraries(${TARGET_NAME} PUBLIC ${SOFACUDA_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${SOFACUDA_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

add_subdirectory(extensions)
