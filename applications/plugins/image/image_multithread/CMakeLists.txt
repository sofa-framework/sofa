cmake_minimum_required(VERSION 3.22)
project(image_multithread VERSION 0.1)

set(HEADER_FILES
        src/image_multithread/MultithreadImageTypes.h
        src/image_multithread/config.h.in
        src/image_multithread/init.h

)

set(SOURCE_FILES
        src/image_multithread/MultithreadImageTypes.cpp
        src/image_multithread/init.cpp
)

sofa_find_package(MultiThreading REQUIRED)
sofa_find_package(image REQUIRED)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES}  ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} image MultiThreading)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
        PACKAGE_NAME ${PROJECT_NAME}
        TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
        PACKAGE_VERSION ${PROJECT_VERSION}
        RELOCATABLE "plugins"
        INCLUDE_SOURCE_DIR "src"
        INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)