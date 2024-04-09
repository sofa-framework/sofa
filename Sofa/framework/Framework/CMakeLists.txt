cmake_minimum_required(VERSION 3.22)
project(Sofa.Framework LANGUAGES CXX)

set(SOFAFRAMEWORK_SOURCE_DIR "src/sofa/framework")

set(SOFAFRAMEWORK_MODULES Sofa.Config Sofa.Type Sofa.Geometry Sofa.Topology Sofa.Helper Sofa.LinearAlgebra Sofa.DefaultType Sofa.Core Sofa.Simulation.Core Sofa.Simulation.Common Sofa.Simulation.Graph Sofa.Testing)
foreach(framework_module ${SOFAFRAMEWORK_MODULES})
    sofa_find_package(${framework_module} REQUIRED)
endforeach()

set(HEADER_FILES
    ${SOFAFRAMEWORK_SOURCE_DIR}/config.h.in
    ${SOFAFRAMEWORK_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFAFRAMEWORK_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFAFRAMEWORK_MODULES})

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
