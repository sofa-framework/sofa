cmake_minimum_required(VERSION 2.8)

project("SofaBaseAnimationLoop")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(headers

	initBaseAnimationLoop.h
    animationloop/MultiStepAnimationLoop.h
    animationloop/MultiTagAnimationLoop.h
	)

set(sources

	initBaseAnimationLoop.cpp
    animationloop/MultiStepAnimationLoop.cpp
    animationloop/MultiTagAnimationLoop.cpp
	)

add_library(${PROJECT_NAME} SHARED ${headers} ${sources})
target_link_libraries(${PROJECT_NAME} SofaTree)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};SOFA_BUILD_BASE_ANIMATION_LOOP")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

