include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initSimpleFEM.h
    BeamFEMForceField.h
    BeamFEMForceField.inl
    HexahedralFEMForceField.h
    HexahedralFEMForceField.inl
    HexahedralFEMForceFieldAndMass.h
    HexahedralFEMForceFieldAndMass.inl
    HexahedronFEMForceField.h
    HexahedronFEMForceField.inl
    HexahedronFEMForceFieldAndMass.h
    HexahedronFEMForceFieldAndMass.inl
    TetrahedralCorotationalFEMForceField.h
    TetrahedralCorotationalFEMForceField.inl
    TetrahedronFEMForceField.h
    TetrahedronFEMForceField.inl
    TriangularAnisotropicFEMForceField.h
    TriangularAnisotropicFEMForceField.inl
    TriangleFEMForceField.h
    TriangleFEMForceField.inl
    TriangularFEMForceField.h
    TriangularFEMForceField.inl
    TriangularFEMForceFieldOptim.h
    TriangularFEMForceFieldOptim.inl
    PoissonContainer.h
    StiffnessContainer.h
    RadiusContainer.h
    LengthContainer.h
    )

set(SOURCE_FILES

    initSimpleFEM.cpp
    BeamFEMForceField.cpp
    HexahedralFEMForceField.cpp
    HexahedralFEMForceFieldAndMass.cpp
    HexahedronFEMForceField.cpp
    HexahedronFEMForceFieldAndMass.cpp
    TetrahedralCorotationalFEMForceField.cpp
    TetrahedronFEMForceField.cpp
    TriangularAnisotropicFEMForceField.cpp
    TriangleFEMForceField.cpp
    TriangularFEMForceField.cpp
    TriangularFEMForceFieldOptim.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SIMPLE_FEM")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaBaseTopology ${SOFA_VISUAL_LIB} newmat)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
