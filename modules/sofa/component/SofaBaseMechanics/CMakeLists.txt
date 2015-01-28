include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initBaseMechanics.h 
    ../container/MappedObject.h 
    ../container/MappedObject.inl 
    ../container/MechanicalObject.h 
    ../container/MechanicalObject.inl 
    ../mass/AddMToMatrixFunctor.h 
    ../mass/DiagonalMass.h 
    ../mass/DiagonalMass.inl 
    ../mass/UniformMass.h 
    ../mass/UniformMass.inl 
    ../mapping/BarycentricMapping.h 
    ../mapping/BarycentricMapping.inl 
    ../mapping/IdentityMapping.h 
    ../mapping/IdentityMapping.inl 
    # ../mapping/SubsetMapping.h 
    # ../mapping/SubsetMapping.inl

    )
    
set(SOURCE_FILES

    ../initBaseMechanics.cpp 
    ../container/MappedObject.cpp 
    ../container/MechanicalObject.cpp 
    ../mass/DiagonalMass.cpp 
    ../mass/UniformMass.cpp 
    ../mapping/BarycentricMapping.cpp 
    ../mapping/IdentityMapping.cpp 
    # ../mapping/SubsetMapping.cpp
    
    )
    
if(SOFA-MISC_SMP)
    list(APPEND HEADER_FILES "../container/MechanicalObjectTasks.inl")
endif()

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_BASE_MECHANICS")
AddSourceDependencies(eigen)
AddLinkerDependencies(SofaBaseTopology SofaBaseLinearSolver)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
