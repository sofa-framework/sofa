include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initSPHFluid.h 
    # ../container/SpatialGridContainer.h 
    # ../container/SpatialGridContainer.inl 
    # ../forcefield/SPHFluidForceField.h 
    # ../forcefield/SPHFluidForceField.inl 
    # ../mapping/SPHFluidSurfaceMapping.h 
    # ../mapping/SPHFluidSurfaceMapping.inl 
    # ../misc/ParticleSink.h 
    # ../misc/ParticleSource.h 
    # ../forcefield/ParticlesRepulsionForceField.h 
    # ../forcefield/ParticlesRepulsionForceField.inl

    )
    
set(SOURCE_FILES

    ../initSPHFluid.cpp 
    ../container/SpatialGridContainer.cpp 
    ../forcefield/SPHFluidForceField.cpp 
    ../mapping/SPHFluidSurfaceMapping.cpp 
    ../misc/ParticleSink.cpp 
    ../misc/ParticleSource.cpp 
    ../forcefield/ParticlesRepulsionForceField.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SPH_FLUID")
AddLinkerDependencies(SofaBaseMechanics)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
