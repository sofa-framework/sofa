cmake_minimum_required(VERSION 3.1)
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

find_package(SofaBase REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_ManualMapping")
target_link_libraries(${PROJECT_NAME} SofaComponentBase)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    RELOCATABLE "plugins"
    )
