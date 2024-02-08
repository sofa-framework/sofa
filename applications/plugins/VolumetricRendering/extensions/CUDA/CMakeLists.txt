cmake_minimum_required(VERSION 3.12)
project(VolumetricRendering.CUDA)

set(HEADER_FILES
    src/VolumetricRendering/CUDA/init.h
    src/VolumetricRendering/CUDA/config.h.in
)

set(SOURCE_FILES
    src/VolumetricRendering/CUDA/init.cpp
)

sofa_find_package(VolumetricRendering REQUIRED)
sofa_find_package(SofaCUDA REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} VolumetricRendering)
target_link_libraries(${PROJECT_NAME} SofaCUDA)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
