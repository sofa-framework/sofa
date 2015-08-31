include(${SOFA_CMAKE_DIR}/preProject.cmake)

##plugin external dependencies
#find_package( REQUIRED)
#include_directories(${})

set(HEADER_FILES
    PersistentContact.h
    PersistentContactBarycentricMapping.h
    PersistentContactBarycentricMapping.inl
    PersistentContactMapping.h
    PersistentContactRigidMapping.h
    PersistentContactRigidMapping.inl
    PersistentFrictionContact.h
    PersistentFrictionContact.inl
    PersistentUnilateralInteractionConstraint.h
    PersistentUnilateralInteractionConstraint.inl
)

set(SOURCE_FILES
    PersistentContactBarycentricMapping.cpp
    PersistentContactMapping.cpp
    PersistentContactRigidMapping.cpp
    PersistentFrictionContact.cpp
    PersistentUnilateralInteractionConstraint.cpp
    initPersistentContact.cpp
)

set(README_FILES PersistentContact.txt)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

AddCompilerDefinitions("SOFA_BUILD_PERSISTENTCONTACT")
AddLinkerDependencies(SofaHelper SofaMeshCollision SofaConstraint )

include(${SOFA_CMAKE_DIR}/postProject.cmake)