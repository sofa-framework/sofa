cmake_minimum_required(VERSION 2.8.12)
project(SofaBaseAnimationLoop)

set(HEADER_FILES
    MultiStepAnimationLoop.h
    MultiTagAnimationLoop.h
    config.h
    initBaseAnimationLoop.h
)

set(SOURCE_FILES
    MultiStepAnimationLoop.cpp
    MultiTagAnimationLoop.cpp
    initBaseAnimationLoop.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCommon)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_BASE_ANIMATION_LOOP")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaBase ${PROJECT_NAME} ${PROJECT_NAME})
