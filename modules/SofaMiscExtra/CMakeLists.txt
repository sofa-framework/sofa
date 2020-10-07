cmake_minimum_required(VERSION 3.12)
project(SofaMiscExtra VERSION 1.0 LANGUAGES CXX)

find_package(SofaMiscTopology REQUIRED)
find_package(SofaGeneral REQUIRED)

set(HEADER_FILES
    ${SOFAMISCOTHER_SRC}/initSofaMiscExtra.h
    ${SOFAMISCOTHER_SRC}/config.h.in
    )

set(SOURCE_FILES
    ${SOFAMISCOTHER_SRC}/initSofaMiscExtra.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAMISCOTHER_SRC}/MeshTetraStuffing.h
    )
list(APPEND SOURCE_FILES
    ${SOFAMISCOTHER_SRC}/MeshTetraStuffing.cpp
    )

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGeneralMeshCollision SofaMiscTopology SofaTopologyMapping)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
    )
