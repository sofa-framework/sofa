include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initDeformable.h
    FastTriangularBendingSprings.h
    FastTriangularBendingSprings.inl
    QuadularBendingSprings.h
    QuadularBendingSprings.inl
    RestShapeSpringsForceField.h
    RestShapeSpringsForceField.inl
    TriangularBendingSprings.h
    TriangularBendingSprings.inl
    TriangularBiquadraticSpringsForceField.h
    TriangularBiquadraticSpringsForceField.inl
    TriangularQuadraticSpringsForceField.h
    TriangularQuadraticSpringsForceField.inl
    TriangularTensorMassForceField.h
    TriangularTensorMassForceField.inl
    FrameSpringForceField.h
    FrameSpringForceField.inl
    MeshSpringForceField.h
    MeshSpringForceField.inl
    QuadBendingSprings.h
    QuadBendingSprings.inl
    RegularGridSpringForceField.h
    RegularGridSpringForceField.inl
    SpringForceField.h
    SpringForceField.inl
    StiffSpringForceField.h
    StiffSpringForceField.inl
    TriangleBendingSprings.h
    TriangleBendingSprings.inl
    VectorSpringForceField.h
    VectorSpringForceField.inl

    )

set(SOURCE_FILES

    initDeformable.cpp
    FastTriangularBendingSprings.cpp
    QuadularBendingSprings.cpp
    RestShapeSpringsForceField.cpp
    TriangularBendingSprings.cpp
    TriangularBiquadraticSpringsForceField.cpp
    TriangularQuadraticSpringsForceField.cpp
    TriangularTensorMassForceField.cpp
    FrameSpringForceField.cpp
    MeshSpringForceField.cpp
    QuadBendingSprings.cpp
    RegularGridSpringForceField.cpp
    SpringForceField.cpp
    StiffSpringForceField.cpp
    TriangleBendingSprings.cpp
    VectorSpringForceField.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_DEFORMABLE")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaBaseTopology)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
