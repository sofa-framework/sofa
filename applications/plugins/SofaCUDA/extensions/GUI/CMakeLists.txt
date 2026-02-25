cmake_minimum_required(VERSION 3.12)
project(SofaCUDA.GUI LANGUAGES CXX CUDA)

set(SOFACUDAGUI_SOURCE_DIR "src/SofaCUDA/GUI")
set(HEADER_FILES
    ${SOFACUDAGUI_SOURCE_DIR}/init.h
    ${SOFACUDAGUI_SOURCE_DIR}/config.h.in
)

set(SOURCE_FILES
    ${SOFACUDAGUI_SOURCE_DIR}/init.cpp
    ${SOFACUDAGUI_SOURCE_DIR}/CudaMouseInteraction.cpp
)

sofa_find_package(SofaCUDA REQUIRED)
sofa_find_package(Sofa.GUI REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaCUDA)
target_link_libraries(${PROJECT_NAME} Sofa.GUI)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
