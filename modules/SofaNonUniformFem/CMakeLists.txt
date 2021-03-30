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
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridGeometryAlgorithms.h
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridGeometryAlgorithms.inl
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridTopologyAlgorithms.h
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridTopologyContainer.h
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridTopologyModifier.h
    ${SOFANONUNIFORMFEM_SRC}/MultilevelHexahedronSetTopologyContainer.h
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedralFEMForceFieldAndMass.h
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedralFEMForceFieldAndMass.inl
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldAndMass.h
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldAndMass.inl
    # ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldDensity.h
    # ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldDensity.inl
    ${SOFANONUNIFORMFEM_SRC}/SparseGridMultipleTopology.h
    ${SOFANONUNIFORMFEM_SRC}/SparseGridRamificationTopology.h
    )
list(APPEND SOURCE_FILES
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridGeometryAlgorithms.cpp
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridTopologyContainer.cpp
    ${SOFANONUNIFORMFEM_SRC}/DynamicSparseGridTopologyModifier.cpp
    ${SOFANONUNIFORMFEM_SRC}/MultilevelHexahedronSetTopologyContainer.cpp
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedralFEMForceFieldAndMass.cpp
    ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldAndMass.cpp
    # ${SOFANONUNIFORMFEM_SRC}/NonUniformHexahedronFEMForceFieldDensity.cpp
    ${SOFANONUNIFORMFEM_SRC}/SparseGridMultipleTopology.cpp
    ${SOFANONUNIFORMFEM_SRC}/SparseGridRamificationTopology.cpp
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

sofa_configure_unity_build(TARGET ${PROJECT_NAME}
                           ACTIVATED "${SOFA_BUILD_WITH_UNITY_BUILD_ENABLED}"
                           BATCHSIZE "${SOFA_UNITY_BUILD_BATCH_SIZE}")

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
    RELOCATABLE "plugins"
)
