include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    config.h
    initMiscFEM.h
    FastTetrahedralCorotationalForceField.h
    FastTetrahedralCorotationalForceField.inl
    TetrahedralTensorMassForceField.h
    TetrahedralTensorMassForceField.inl
	StandardTetrahedralFEMForceField.h
	StandardTetrahedralFEMForceField.inl
	BoyceAndArruda.h
	Costa.h
	HyperelasticMaterial.h
	MooneyRivlin.h
	PlasticMaterial.h
	STVenantKirchhoff.h
	VerondaWestman.h

    )

set(SOURCE_FILES

    initMiscFEM.cpp
    FastTetrahedralCorotationalForceField.cpp
    TetrahedralTensorMassForceField.cpp
 	StandardTetrahedralFEMForceField.cpp
	PlasticMaterial.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_FEM" )
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaSimpleFem )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
