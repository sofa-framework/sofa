cmake_minimum_required(VERSION 3.1)
project(SofaGeneralAnimationLoop)

set(HEADER_FILES
    config.h
    initGeneralAnimationLoop.h
    )

set(SOURCE_FILES
    initGeneralAnimationLoop.cpp
    )

list(APPEND HEADER_FILES
    MultiStepAnimationLoop.h
    MultiTagAnimationLoop.h )

list(APPEND SOURCE_FILES
    MultiStepAnimationLoop.cpp
    MultiTagAnimationLoop.cpp )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_GENERAL_ANIMATION_LOOP")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaBase ${PROJECT_NAME} ${PROJECT_NAME})
