include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initSPHFluid.h
    SpatialGridContainer.h
    SpatialGridContainer.inl
    SPHFluidForceField.h
    SPHFluidForceField.inl
    SPHFluidSurfaceMapping.h
    SPHFluidSurfaceMapping.inl
    ParticleSink.h
    ParticleSource.h
    ParticlesRepulsionForceField.h
    ParticlesRepulsionForceField.inl

    )

set(SOURCE_FILES

    initSPHFluid.cpp
    SpatialGridContainer.cpp
    SPHFluidForceField.cpp
    SPHFluidSurfaceMapping.cpp
    ParticleSink.cpp
    ParticleSource.cpp
    ParticlesRepulsionForceField.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SPH_FLUID")
AddLinkerDependencies(SofaBaseMechanics)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
