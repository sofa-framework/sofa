cmake_minimum_required(VERSION 3.22)
project(SofaGeneralTopology LANGUAGES CXX)

set(SOFAGENERALTOPOLOGY_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALTOPOLOGY_SRC}/config.h.in
    ${SOFAGENERALTOPOLOGY_SRC}/initSofaGeneralTopology.h
    )

set(SOURCE_FILES
    ${SOFAGENERALTOPOLOGY_SRC}/initSofaGeneralTopology.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Topology.Container.Grid and Sofa.Component.Topology.Container.Constant instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Topology.Container.Constant REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Grid REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Constant Sofa.Component.Topology.Container.Grid)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)

