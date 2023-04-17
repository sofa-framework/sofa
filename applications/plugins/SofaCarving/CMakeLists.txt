cmake_minimum_required(VERSION 3.12)
project(SofaCarving VERSION 0.1 LANGUAGES CXX)

set(SOFACARVING_SRC src/SofaCarving)

set(HEADER_FILES
    ${SOFACARVING_SRC}/config.h.in
    ${SOFACARVING_SRC}/init.h
    ${SOFACARVING_SRC}/CarvingManager.h
)

set(SOURCE_FILES
    ${SOFACARVING_SRC}/init.cpp
    ${SOFACARVING_SRC}/CarvingManager.cpp
)

set(README_FILES SofaCarving.txt)

find_package(Sofa.GUI.Component REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

target_link_libraries(${PROJECT_NAME} Sofa.GUI.Component)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )

if(SOFA_BUILD_TESTS)
    add_subdirectory(SofaCarving_test)
endif()
