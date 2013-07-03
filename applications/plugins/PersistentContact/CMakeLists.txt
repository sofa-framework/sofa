include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES

    PersistentContact.h 
	PersistentContactMapping.h 
	PersistentFrictionContact.h 
	PersistentFrictionContact.inl 
	PersistentUnilateralInteractionConstraint.h 
	PersistentUnilateralInteractionConstraint.inl 
	PersistentContactBarycentricMapping.h 
	PersistentContactBarycentricMapping.inl 
	PersistentContactRigidMapping.h 
	PersistentContactRigidMapping.inl

	)

set(SOURCE_FILES

    PersistentContactMapping.cpp 
	PersistentFrictionContact.cpp 
	PersistentUnilateralInteractionConstraint.cpp 
	PersistentContactBarycentricMapping.cpp 
	PersistentContactRigidMapping.cpp 
    initPersistentContact.cpp

	)

set(README_FILES

	PersistentContact.txt)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_PERSISTENTCONTACT" "SOFA_DEV") # this plugin won't build without SOFA_DEV
AddLinkerDependencies(SofaHelper SofaMeshCollision SofaConstraint )

include(${SOFA_CMAKE_DIR}/postProject.cmake)