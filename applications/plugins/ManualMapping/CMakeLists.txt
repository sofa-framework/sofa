cmake_minimum_required(VERSION 3.22)
project(ManualMapping VERSION 0.1)


set(HEADER_FILES
    ManualLinearMapping.h
    ManualLinearMapping.inl
    config.h
)

set(SOURCE_FILES
    ManualLinearMapping.cpp
    initManualMapping.cpp
)

find_package(Sofa.Core REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} Sofa.Core)
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_${PROJECT_NAME}")

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    RELOCATABLE "plugins"
    )
