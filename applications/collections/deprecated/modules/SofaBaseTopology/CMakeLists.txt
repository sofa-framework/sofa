cmake_minimum_required(VERSION 3.22)
project(SofaBaseTopology LANGUAGES CXX)

set(SOFABASETOPOLOGY_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASETOPOLOGY_SRC}/config.h.in
    ${SOFABASETOPOLOGY_SRC}/initSofaBaseTopology.h
)

set(SOURCE_FILES
    ${SOFABASETOPOLOGY_SRC}/initSofaBaseTopology.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.Component.Topology.Container.Grid, Sofa.Component.Topology.Container.Constant and Sofa.Component.Topology.Container.Dynamic instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Topology.Container.Constant REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Grid REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Constant Sofa.Component.Topology.Container.Grid Sofa.Component.Topology.Container.Dynamic)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "collections"
)
