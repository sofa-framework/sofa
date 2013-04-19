cmake_minimum_required(VERSION 2.8)

project("SofaMiscFem")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initMiscFEM.h 
    ../forcefield/FastTetrahedralCorotationalForceField.h 
    ../forcefield/FastTetrahedralCorotationalForceField.inl 
    ../forcefield/TetrahedralTensorMassForceField.h 
    ../forcefield/TetrahedralTensorMassForceField.inl 

    )
    
set(SOURCE_FILES

    ../initMiscFEM.cpp 
    ../forcefield/FastTetrahedralCorotationalForceField.cpp 
    ../forcefield/TetrahedralTensorMassForceField.cpp 
 
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_MISC_FEM" )
set(LINKER_DEPENDENCIES SofaSimpleFem )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
