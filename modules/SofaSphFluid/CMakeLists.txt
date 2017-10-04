cmake_minimum_required(VERSION 3.1)
project(SofaSphFluid)

set(HEADER_FILES
    ParticleSink.h
    ParticleSource.h
    ParticlesRepulsionForceField.h
    ParticlesRepulsionForceField.inl
    SPHFluidForceField.h
    SPHFluidForceField.inl
    SPHFluidSurfaceMapping.h
    SPHFluidSurfaceMapping.inl
    SpatialGridContainer.h
    SpatialGridContainer.inl
    config.h
    initSPHFluid.h
)

set(SOURCE_FILES
    ParticleSink.cpp
    ParticleSource.cpp
    ParticlesRepulsionForceField.cpp
    SPHFluidForceField.cpp
    SPHFluidSurfaceMapping.cpp
    SpatialGridContainer.cpp
    initSPHFluid.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBaseTopology SofaBaseMechanics)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SPH_FLUID")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaAdvanced ${PROJECT_NAME} ${PROJECT_NAME})
