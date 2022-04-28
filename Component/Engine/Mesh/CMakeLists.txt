cmake_minimum_required(VERSION 3.12)
project(Sofa.Component.Engine.Mesh LANGUAGES CXX)

set(SOFACOMPONENTENGINEMESH_SOURCE_DIR "src/sofa/component/engine/mesh")

set(HEADER_FILES
    ${SOFACOMPONENTENGINEMESH_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTENGINEMESH_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTENGINEMESH_SOURCE_DIR}/init.cpp    

)

sofa_find_package(SofaSimulationCore REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationCore)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
