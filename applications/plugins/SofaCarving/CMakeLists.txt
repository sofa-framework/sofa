cmake_minimum_required(VERSION 3.12)
project(SofaCarving VERSION 0.1)

set(HEADER_FILES
    CarvingManager.h
    config.h
)

set(SOURCE_FILES
    CarvingManager.cpp
    initSofaCarving.cpp
)

set(README_FILES SofaCarving.txt)

sofa_find_package(SofaUserInteraction REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

target_link_libraries(${PROJECT_NAME} SofaUserInteraction)

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    RELOCATABLE "plugins"
    )

if(SOFA_BUILD_TESTS)
    find_package(SofaTest QUIET)
    if(SofaTest_FOUND)
        add_subdirectory(SofaCarving_test)
    endif()
endif()
