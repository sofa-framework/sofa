cmake_minimum_required(VERSION 3.12)
project(SofaBaseMechanics LANGUAGES CXX)

set(SOFABASEMECHANICS_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFABASEMECHANICS_SRC}/config.h.in
    ${SOFABASEMECHANICS_SRC}/initSofaBaseMechanics.h
    ${SOFABASEMECHANICS_SRC}/AddMToMatrixFunctor.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMapping.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMapping.inl
    ${SOFABASEMECHANICS_SRC}/DiagonalMass.h
    ${SOFABASEMECHANICS_SRC}/DiagonalMass.inl
    ${SOFABASEMECHANICS_SRC}/IdentityMapping.h
    ${SOFABASEMECHANICS_SRC}/IdentityMapping.inl
    ${SOFABASEMECHANICS_SRC}/MappedObject.h
    ${SOFABASEMECHANICS_SRC}/MappedObject.inl
    ${SOFABASEMECHANICS_SRC}/MechanicalObject.h
    ${SOFABASEMECHANICS_SRC}/MechanicalObject.inl
    ${SOFABASEMECHANICS_SRC}/SubsetMapping.h
    ${SOFABASEMECHANICS_SRC}/SubsetMapping.inl
    ${SOFABASEMECHANICS_SRC}/UniformMass.h
    ${SOFABASEMECHANICS_SRC}/UniformMass.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapper.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapper.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/TopologyBarycentricMapper.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/TopologyBarycentricMapper.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperMeshTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperMeshTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperRegularGridTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperRegularGridTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperSparseGridTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperSparseGridTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTopologyContainer.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTopologyContainer.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperEdgeSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperEdgeSetTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTriangleSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTriangleSetTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperQuadSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperQuadSetTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.inl
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.h
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.inl
)

set(SOURCE_FILES
    ${SOFABASEMECHANICS_SRC}/initSofaBaseMechanics.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMapping.cpp
    ${SOFABASEMECHANICS_SRC}/DiagonalMass.cpp
    ${SOFABASEMECHANICS_SRC}/IdentityMapping.cpp
    ${SOFABASEMECHANICS_SRC}/MappedObject.cpp
    ${SOFABASEMECHANICS_SRC}/MechanicalObject.cpp
    ${SOFABASEMECHANICS_SRC}/SubsetMapping.cpp
    ${SOFABASEMECHANICS_SRC}/UniformMass.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapper.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/TopologyBarycentricMapper.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperMeshTopology.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperRegularGridTopology.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperSparseGridTopology.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTopologyContainer.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperEdgeSetTopology.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTriangleSetTopology.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperQuadSetTopology.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperTetrahedronSetTopology.cpp
    ${SOFABASEMECHANICS_SRC}/BarycentricMappers/BarycentricMapperHexahedronSetTopology.cpp
)

sofa_find_package(SofaEigen2Solver REQUIRED)
sofa_find_package(SofaBaseTopology REQUIRED)
sofa_find_package(SofaBaseLinearSolver REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaBaseTopology SofaEigen2Solver SofaBaseLinearSolver)

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
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
#cmake_dependent_option(SOFABASEMECHANICS_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
#if(SOFABASEMECHANICS_BUILD_TESTS)
#    enable_testing()
#    add_subdirectory(${PROJECT_NAME}_test)
#endif()
