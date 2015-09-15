cmake_minimum_required(VERSION 2.8.12)
project(SofaComponentAdvanced)

set(HEADER_FILES
    config.h
    initComponentAdvanced.h
)

set(SOURCE_FILES
    initComponentAdvanced.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaEulerianFluid SofaNonUniformFem SofaSphFluid SofaVolumetricData)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_COMPONENT_ADVANCED")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaAdvanced_install(${PROJECT_NAME})
