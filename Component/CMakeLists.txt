cmake_minimum_required(VERSION 3.12)
project(Sofa.Component LANGUAGES CXX)

# force Compat if already not activated
if(NOT TARGET Sofa.Component.Compat)
	add_subdirectory(Compat)
endif()

set(SOFACOMPONENT_SOURCE_DIR "src/sofa/component")

set(SOFACOMPONENT_DIRS ODESolver IO Playback SceneUtility) # LinearSolver Topology PhysicalModel Constraint AnimationLoop Mapping Container Collision Engine Visual UI Controller) # Device
set(SOFACOMPONENT_TARGETS)
foreach(component_dir ${SOFACOMPONENT_DIRS})
	sofang_add_component_subdirectory(${component_dir} ${PROJECT_NAME}.${component_dir})
	list(APPEND SOFACOMPONENT_TARGETS ${PROJECT_NAME}.${component_dir})
endforeach()

set(HEADER_FILES
    ${SOFACOMPONENT_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENT_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENT_SOURCE_DIR}/init.cpp
)

set(TARGET_NAME ${PROJECT_NAME})

add_library(${TARGET_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${TARGET_NAME} PUBLIC ${SOFACOMPONENT_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
