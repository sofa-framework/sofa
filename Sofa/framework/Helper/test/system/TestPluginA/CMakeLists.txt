cmake_minimum_required(VERSION 3.12)
project(TestPluginA VERSION 0.7)


set(HEADER_FILES
    ComponentA.h
    ComponentB.h
    TestPluginA.h
)

set(SOURCE_FILES
    ComponentA.cpp
    ComponentB.cpp
    initTestPluginA.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Core Sofa.DefaultType)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
