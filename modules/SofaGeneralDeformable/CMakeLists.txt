cmake_minimum_required(VERSION 3.12)
project(SofaGeneralDeformable LANGUAGES CXX)


set(SOFAGENERALDEFORMABLE_SRC "src/${PROJECT_NAME}")

set(HEADER_FILES
    ${SOFAGENERALDEFORMABLE_SRC}/config.h.in
    ${SOFAGENERALDEFORMABLE_SRC}/initSofaGeneralDeformable.h
    )

set(SOURCE_FILES
    ${SOFAGENERALDEFORMABLE_SRC}/initSofaGeneralDeformable.cpp
    )

list(APPEND HEADER_FILES
    ${SOFAGENERALDEFORMABLE_SRC}/FastTriangularBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/FastTriangularBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/FrameSpringForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/FrameSpringForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/QuadBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/QuadBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/QuadularBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/QuadularBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/RegularGridSpringForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/RegularGridSpringForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangleBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangleBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBendingSprings.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBendingSprings.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBiquadraticSpringsForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBiquadraticSpringsForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularQuadraticSpringsForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularQuadraticSpringsForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularTensorMassForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularTensorMassForceField.inl
    ${SOFAGENERALDEFORMABLE_SRC}/VectorSpringForceField.h
    ${SOFAGENERALDEFORMABLE_SRC}/VectorSpringForceField.inl
    )
list(APPEND SOURCE_FILES
    ${SOFAGENERALDEFORMABLE_SRC}/FastTriangularBendingSprings.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/FrameSpringForceField.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/QuadBendingSprings.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/QuadularBendingSprings.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/RegularGridSpringForceField.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/TriangleBendingSprings.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBendingSprings.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularBiquadraticSpringsForceField.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularQuadraticSpringsForceField.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/TriangularTensorMassForceField.cpp
    ${SOFAGENERALDEFORMABLE_SRC}/VectorSpringForceField.cpp
    )


find_package(SofaBase REQUIRED) # SofaBaseTopology
find_package(SofaCommon REQUIRED) # SofaDeformable

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaDeformable)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
# cmake_dependent_option(SOFAGENERALDEFORMABLE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
# if(SOFAGENERALDEFORMABLE_BUILD_TESTS)
#     enable_testing()
#     add_subdirectory(${PROJECT_NAME}_test)
# endif()

