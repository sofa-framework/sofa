cmake_minimum_required(VERSION 3.12)
project(SofaNonUniformFem LANGUAGES CXX)

sofa_find_package(SofaBase REQUIRED)
sofa_find_package(SofaSimpleFem REQUIRED)
sofa_find_package(SofaGeneralSimpleFem REQUIRED)

set(SOFANONUNIFORMFEM_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFANONUNIFORMFEM_SRC}/config.h.in
    ${SOFANONUNIFORMFEM_SRC}/initSofaNonUniformFem.h
    )

set(SOURCE_FILES
    ${SOFANONUNIFORMFEM_SRC}/initSofaNonUniformFem.cpp
    )

list(APPEND HEADER_FILES
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedralFEMForceFieldAndMass.h
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedralFEMForceFieldAndMass.inl
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldAndMass.h
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldAndMass.inl
    )
list(APPEND SOURCE_FILES
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedralFEMForceFieldAndMass.cpp
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldAndMass.cpp
    )

sofa_find_package(SofaDenseSolver QUIET)
if(SofaDenseSolver_FOUND)
    list(APPEND HEADER_FILES
        ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMForceFieldAndMass.h
        ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMForceFieldAndMass.inl
        ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMMapping.h
        ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMMapping.inl
    )
    list(APPEND SOURCE_FILES
        ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMForceFieldAndMass.cpp
        ${SOFANONUNIFORMFEM_SRC}/HexahedronCompositeFEMMapping.cpp
    )
else()
    message(STATUS "SofaDenseSolver not found, disabling HexahedronCompositeFEMForceFieldAndMass compilation.")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimpleFem SofaGeneralSimpleFem SofaBaseTopology SofaBaseMechanics)
if(SofaDenseSolver_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaDenseSolver)
endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

message(WARNING "${PROJECT_NAME} module is being deprecated. It will be removed at v23.06. You may use Sofa.Component.Topology.Container.Grid and Sofa.Component.Topology.Container.Dynamic instead.")

# forward to the new sofang module
sofa_find_package(Sofa.Component.Topology.Container.Grid REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Topology.Container.Grid Sofa.Component.Topology.Container.Dynamic)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
