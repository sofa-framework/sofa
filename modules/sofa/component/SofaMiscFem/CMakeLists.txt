include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMiscFEM.h 
 #    ../forcefield/FastTetrahedralCorotationalForceField.h 
 #    ../forcefield/FastTetrahedralCorotationalForceField.inl 
 #    ../forcefield/TetrahedralTensorMassForceField.h 
 #    ../forcefield/TetrahedralTensorMassForceField.inl 
	# ../forcefield/StandardTetrahedralFEMForceField.h
	# ../forcefield/StandardTetrahedralFEMForceField.inl
	# ../fem/material/BoyceAndArruda.h
	# ../fem/material/Costa.h
	# ../fem/material/HyperelasticMaterial.h
	# ../fem/material/MooneyRivlin.h
	# ../fem/material/PlasticMaterial.h
	# ../fem/material/STVenantKirchhoff.h
	# ../fem/material/VerondaWestman.h

    )
    
set(SOURCE_FILES

    ../initMiscFEM.cpp 
 #    ../forcefield/FastTetrahedralCorotationalForceField.cpp 
 #    ../forcefield/TetrahedralTensorMassForceField.cpp 
 # 	../forcefield/StandardTetrahedralFEMForceField.cpp
	# ../fem/material/PlasticMaterial.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_MISC_FEM" )
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaSimpleFem )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
