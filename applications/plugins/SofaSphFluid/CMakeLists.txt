cmake_minimum_required(VERSION 3.1)
project(SofaSphFluid VERSION 1.0)

find_package(SofaBase REQUIRED)
 
set(PLUGIN_SPH_SRC_DIR src/SofaSphFluid)
set(HEADER_FILES
    ${PLUGIN_SPH_SRC_DIR}/config.h.in
    ${PLUGIN_SPH_SRC_DIR}/ParticleSink.h
    ${PLUGIN_SPH_SRC_DIR}/ParticleSource.h
    ${PLUGIN_SPH_SRC_DIR}/ParticlesRepulsionForceField.h
    ${PLUGIN_SPH_SRC_DIR}/ParticlesRepulsionForceField.inl
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidForceField.h
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidForceField.inl
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidSurfaceMapping.h
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidSurfaceMapping.inl
    ${PLUGIN_SPH_SRC_DIR}/SpatialGridContainer.h
    ${PLUGIN_SPH_SRC_DIR}/SpatialGridContainer.inl    
)

set(SOURCE_FILES
    ${PLUGIN_SPH_SRC_DIR}/initSPHFluid.cpp
    ${PLUGIN_SPH_SRC_DIR}/ParticleSink.cpp
    ${PLUGIN_SPH_SRC_DIR}/ParticleSource.cpp
    ${PLUGIN_SPH_SRC_DIR}/ParticlesRepulsionForceField.cpp
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidForceField.cpp
    ${PLUGIN_SPH_SRC_DIR}/SPHFluidSurfaceMapping.cpp
    ${PLUGIN_SPH_SRC_DIR}/SpatialGridContainer.cpp
)

set(README_FILES
    README.md
)

# Create the plugin library.
add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${README_FILES})

# Set define dllimport/dllexport mechanism on Windows.
target_compile_definitions(${PROJECT_NAME} PRIVATE "-DSOFA_BUILD_SPH_FLUID")

# Link the plugin library to its dependencies (other libraries).
target_link_libraries(${PROJECT_NAME} SofaBaseTopology SofaBaseMechanics)


target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")


sofa_generate_package(
    NAME ${PROJECT_NAME}
    TARGETS ${PROJECT_NAME}
    VERSION ${PROJECT_VERSION}
    INCLUDE_SOURCE_DIR ${PLUGIN_SPH_SRC_DIR}
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    RELOCATABLE "plugins"
    )
