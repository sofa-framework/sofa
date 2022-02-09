cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Topology LANGUAGES CXX)

set(SOFACOMPONENTTOPOLOGY_SOURCE_DIR "src/sofa/component/topology")

set(SOFACOMPONENTTOPOLOGY_DIRS Container Mapping Utility)
set(SOFACOMPONENTTOPOLOGY_TARGETS)
foreach(topology_dir ${SOFACOMPONENTTOPOLOGY_DIRS})
	sofang_add_component_subdirectory(${topology_dir} ${PROJECT_NAME}.${topology_dir})
	list(APPEND SOFACOMPONENTTOPOLOGY_TARGETS ${PROJECT_NAME}.${topology_dir})
endforeach()

set(HEADER_FILES
    ${SOFACOMPONENTTOPOLOGY_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTTOPOLOGY_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTTOPOLOGY_SOURCE_DIR}/init.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC ${SOFACOMPONENTTOPOLOGY_TARGETS})

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
