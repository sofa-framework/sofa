cmake_minimum_required(VERSION 3.1)
project(SofaComponentAdvanced)

set(HEADER_FILES
    config.h
    initComponentAdvanced.h
)

set(SOURCE_FILES
    initComponentAdvanced.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaNonUniformFem)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_COMPONENT_ADVANCED")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaAdvanced ${PROJECT_NAME} ${PROJECT_NAME})
