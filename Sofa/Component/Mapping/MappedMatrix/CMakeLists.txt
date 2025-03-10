cmake_minimum_required(VERSION 3.22)
project(Sofa.Component.Mapping.MappedMatrix LANGUAGES CXX)

set(SOFACOMPONENTMAPPINGMAPPEDMATRIX_SOURCE_DIR "src/sofa/component/mapping/mappedmatrix")

set(HEADER_FILES
    ${SOFACOMPONENTMAPPINGMAPPEDMATRIX_SOURCE_DIR}/config.h.in
    ${SOFACOMPONENTMAPPINGMAPPEDMATRIX_SOURCE_DIR}/init.h
)

set(SOURCE_FILES
    ${SOFACOMPONENTMAPPINGMAPPEDMATRIX_SOURCE_DIR}/init.cpp
)

sofa_find_package(Sofa.Simulation.Core REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Core)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)
