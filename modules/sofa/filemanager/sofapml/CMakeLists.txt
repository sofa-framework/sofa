cmake_minimum_required(VERSION 2.8)

project("SofaPml")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    sofapml.h 
    PMLBody.h 
    PMLRigidBody.h 
    PMLFemForceField.h 
    PMLStiffSpringForceField.h 
    PMLInteractionForceField.h 
    PMLMappedBody.h 
    PMLReader.h 
    LMLConstraint.h 
    LMLConstraint.inl 
    LMLForce.h 
    LMLForce.inl 
    LMLReader.h

	)

set(SOURCE_FILES

    PMLBody.cpp 
    PMLRigidBody.cpp 
    PMLFemForceField.cpp 
    PMLStiffSpringForceField.cpp 
    PMLInteractionForceField.cpp 
    PMLMappedBody.cpp 
    PMLReader.cpp 
    LMLConstraint.cpp 
    LMLForce.cpp 
    LMLReader.cpp 

	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_FILEMANAGER_PML")
set(LINKER_DEPENDENCIES SofaSimulationCommon )

include(${SOFA_CMAKE_DIR}/postProject.cmake)