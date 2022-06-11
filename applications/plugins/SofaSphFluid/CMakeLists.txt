cmake_minimum_required(VERSION 3.12)
project(SofaSphFluid VERSION 1.0)

find_package(Sofa.Core REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Constant REQUIRED)
sofa_find_package(Sofa.Component.Topology.Container.Dynamic REQUIRED)
sofa_find_package(Sofa.Component.Collision.Geometry REQUIRED)
sofa_find_package(Sofa.Component.StateContainer REQUIRED)
sofa_find_package(Sofa.GL QUIET)
 
set(PLUGIN_SPH_SRC_DIR src/SofaSphFluid)
set(HEADER_FILES
    ${PLUGIN_SPH_SRC_DIR}/config.h.in
    ${PLUGIN_SPH_SRC_DIR}/ParticleSink.h
    ${PLUGIN_SPH_SRC_DIR}/ParticleSink.inl
    ${PLUGIN_SPH_SRC_DIR}/ParticleSource.h
	${PLUGIN_SPH_SRC_DIR}/ParticleSource.inl
    ${PLUGIN_SPH_SRC_DIR}/ParticlesRepulsionForceField.h
    ${PLUGIN_SPH_SRC_DIR}/ParticlesRepulsionForceField.inl
    ${PLUGIN_SPH_SRC_DIR}/SPHKernel.h
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidForceField.h
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidForceField.inl
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidSurfaceMapping.h
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidSurfaceMapping.inl
    ${PLUGIN_SPH_SRC_DIR}/SpatialGridContainer.h
    ${PLUGIN_SPH_SRC_DIR}/SpatialGridContainer.inl
    ${PLUGIN_SPH_SRC_DIR}/SpatialGridPointModel.h
)

set(SOURCE_FILES
    ${PLUGIN_SPH_SRC_DIR}/initSPHFluid.cpp
    ${PLUGIN_SPH_SRC_DIR}/ParticleSink.cpp
    ${PLUGIN_SPH_SRC_DIR}/ParticleSource.cpp
    ${PLUGIN_SPH_SRC_DIR}/ParticlesRepulsionForceField.cpp
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidForceField.cpp
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidSurfaceMapping.cpp
    ${PLUGIN_SPH_SRC_DIR}/SpatialGridContainer.cpp
    ${PLUGIN_SPH_SRC_DIR}/SpatialGridPointModel.cpp
)

if(Sofa.GL_FOUND)
    list(APPEND HEADER_FILES
            ${PLUGIN_SPH_SRC_DIR}/OglFluidModel.h
            ${PLUGIN_SPH_SRC_DIR}/OglFluidModel.inl
        )

    list(APPEND SOURCE_FILES
            ${PLUGIN_SPH_SRC_DIR}/OglFluidModel.cpp
        )
endif()

set(README_FILES
    README.md
)

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

# Set define dllimport/dllexport mechanism on Windows.
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SPH_FLUID")

# Link the plugin library to its dependencies (other libraries).
target_link_libraries(${PROJECT_NAME} Sofa.Core Sofa.Component.Topology.Container.Dynamic Sofa.Component.Topology.Container.Constant Sofa.Component.Collision.Geometry Sofa.Component.StateContainer)

if(Sofa.GL_FOUND)
    target_link_libraries(${PROJECT_NAME} Sofa.GL)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )
