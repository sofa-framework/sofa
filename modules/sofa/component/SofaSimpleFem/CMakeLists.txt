include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initSimpleFEM.h 
    # ../forcefield/BeamFEMForceField.h 
    # ../forcefield/BeamFEMForceField.inl 
    # ../forcefield/HexahedralFEMForceField.h 
    # ../forcefield/HexahedralFEMForceField.inl 
    # ../forcefield/HexahedralFEMForceFieldAndMass.h 
    # ../forcefield/HexahedralFEMForceFieldAndMass.inl 
    ../forcefield/HexahedronFEMForceField.h 
    ../forcefield/HexahedronFEMForceField.inl 
    # ../forcefield/HexahedronFEMForceFieldAndMass.h 
    # ../forcefield/HexahedronFEMForceFieldAndMass.inl 
    # ../forcefield/TetrahedralCorotationalFEMForceField.h 
    # ../forcefield/TetrahedralCorotationalFEMForceField.inl 
    ../forcefield/TetrahedronFEMForceField.h 
    ../forcefield/TetrahedronFEMForceField.inl 
    # ../forcefield/TriangularAnisotropicFEMForceField.h 
    # ../forcefield/TriangularAnisotropicFEMForceField.inl 
    ../forcefield/TriangleFEMForceField.h 
    ../forcefield/TriangleFEMForceField.inl 
    # ../forcefield/TriangularFEMForceField.h 
    # ../forcefield/TriangularFEMForceField.inl 
    # ../forcefield/TriangularFEMForceFieldOptim.h 
    # ../forcefield/TriangularFEMForceFieldOptim.inl 
    # ../container/PoissonContainer.h 
    # ../container/StiffnessContainer.h 
    # ../container/RadiusContainer.h 
    # ../container/LengthContainer.h 
    )
    
set(SOURCE_FILES

    ../initSimpleFEM.cpp 
    # ../forcefield/BeamFEMForceField.cpp 
    # ../forcefield/HexahedralFEMForceField.cpp 
    # ../forcefield/HexahedralFEMForceFieldAndMass.cpp 
    ../forcefield/HexahedronFEMForceField.cpp 
    # ../forcefield/HexahedronFEMForceFieldAndMass.cpp 
    # ../forcefield/TetrahedralCorotationalFEMForceField.cpp 
    ../forcefield/TetrahedronFEMForceField.cpp 
    # ../forcefield/TriangularAnisotropicFEMForceField.cpp 
    ../forcefield/TriangleFEMForceField.cpp 
    # ../forcefield/TriangularFEMForceField.cpp 
    # ../forcefield/TriangularFEMForceFieldOptim.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SIMPLE_FEM")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaBaseTopology ${SOFA_VISUAL_LIB} newmat)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
