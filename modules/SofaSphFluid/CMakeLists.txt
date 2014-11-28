cmake_minimum_required(VERSION 2.8.11)
project(SofaSphFluid)

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

target_link_libraries( ${PROJECT_NAME} SofaBaseTopology SofaBaseMechanics ) 
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAADVANCED_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAADVANCED_DEPENDENCY_DIR}>")

set( COMPILER_FLAGS "-DSOFA_BUILD_SPH_FLUID")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAADVANCED_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaAdvancedTargets
        RUNTIME DESTINATION "${SOFAADVANCED_INSTALL_BIN_DIR}" COMPONENT SofaAdvanced_libraries
        LIBRARY DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}" COMPONENT SofaAdvanced_libraries
        ARCHIVE DESTINATION "${SOFAADVANCED_INSTALL_LIB_DIR}" COMPONENT SofaAdvanced_libraries
        PUBLIC_HEADER DESTINATION "${SOFAADVANCED_INSTALL_INCLUDE_DIR}/SofaSphFluid" COMPONENT SofaAdvanced_libraries
        )
