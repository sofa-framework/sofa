cmake_minimum_required(VERSION 2.8.12)
project(SofaDeformable)

set(HEADER_FILES
    FastTriangularBendingSprings.h
    FastTriangularBendingSprings.inl
    FrameSpringForceField.h
    FrameSpringForceField.inl
    MeshSpringForceField.h
    MeshSpringForceField.inl
    QuadBendingSprings.h
    QuadBendingSprings.inl
    QuadularBendingSprings.h
    QuadularBendingSprings.inl
    RegularGridSpringForceField.h
    RegularGridSpringForceField.inl
    RestShapeSpringsForceField.h
    RestShapeSpringsForceField.inl
    SpringForceField.h
    SpringForceField.inl
    StiffSpringForceField.h
    StiffSpringForceField.inl
    TriangleBendingSprings.h
    TriangleBendingSprings.inl
    TriangularBendingSprings.h
    TriangularBendingSprings.inl
    TriangularBiquadraticSpringsForceField.h
    TriangularBiquadraticSpringsForceField.inl
    TriangularQuadraticSpringsForceField.h
    TriangularQuadraticSpringsForceField.inl
    TriangularTensorMassForceField.h
    TriangularTensorMassForceField.inl
    VectorSpringForceField.h
    VectorSpringForceField.inl
    config.h
    initDeformable.h
)

set(SOURCE_FILES
    FastTriangularBendingSprings.cpp
    FrameSpringForceField.cpp
    MeshSpringForceField.cpp
    QuadBendingSprings.cpp
    QuadularBendingSprings.cpp
    RegularGridSpringForceField.cpp
    RestShapeSpringsForceField.cpp
    SpringForceField.cpp
    StiffSpringForceField.cpp
    TriangleBendingSprings.cpp
    TriangularBendingSprings.cpp
    TriangularBiquadraticSpringsForceField.cpp
    TriangularQuadraticSpringsForceField.cpp
    TriangularTensorMassForceField.cpp
    VectorSpringForceField.cpp
    initDeformable.cpp
)


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_DEFORMABLE")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

SofaCommon_install(${PROJECT_NAME})
