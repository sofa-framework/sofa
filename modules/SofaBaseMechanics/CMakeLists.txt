include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initBaseMechanics.h
    MappedObject.h
    MappedObject.inl
    MechanicalObject.h
    MechanicalObject.inl
    AddMToMatrixFunctor.h
    DiagonalMass.h
    DiagonalMass.inl
    UniformMass.h
    UniformMass.inl
    BarycentricMapping.h
    BarycentricMapping.inl
    IdentityMapping.h
    IdentityMapping.inl
    SubsetMapping.h
    SubsetMapping.inl

    )

set(SOURCE_FILES

    initBaseMechanics.cpp
    MappedObject.cpp
    MechanicalObject.cpp
    DiagonalMass.cpp
    UniformMass.cpp
    BarycentricMapping.cpp
    IdentityMapping.cpp
    SubsetMapping.cpp

    )

if(SOFA-MISC_SMP)
    list(APPEND HEADER_FILES "MechanicalObjectTasks.inl")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_MECHANICS")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaBaseTopology SofaBaseLinearSolver)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
