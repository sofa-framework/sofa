cmake_minimum_required(VERSION 3.22)
project(FailingPlugin VERSION 0.7)


set(HEADER_FILES
    ComponentFailingPlugin.h
    FailingPlugin.h
)

set(SOURCE_FILES
    ComponentFailingPlugin.cpp
    init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Core Sofa.DefaultType)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
