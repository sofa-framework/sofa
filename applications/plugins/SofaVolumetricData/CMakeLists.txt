cmake_minimum_required(VERSION 3.1)
project(SofaVolumetricData)

set(SOFAVOLUMETRICDATA_VERSION 1.0)

find_package(SofaDistanceGrid REQUIRED)
find_package(SofaImplicitField REQUIRED)

set(HEADER_FILES
    config.h.in
    initVolumetricData.h
    DistanceGrid.h
    FFDDistanceGridDiscreteIntersection.h
    FFDDistanceGridDiscreteIntersection.inl
    RigidDistanceGridDiscreteIntersection.h
    RigidDistanceGridDiscreteIntersection.inl
    DistanceGridCollisionModel.h
    DistanceGridForceField.h
    DistanceGridForceField.inl
    ImplicitSurfaceContainer.h
    ImplicitSurfaceMapping.h
    ImplicitSurfaceMapping.inl
    InterpolatedImplicitSurface.h
    InterpolatedImplicitSurface.inl
)

set(SOURCE_FILES
    initVolumetricData.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision ${SOFAADVANCED_DEPENDENCY_LINK})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaUserInteraction)

## Install rules for the library and headers; CMake package configurations files
sofa_generate_package(
    NAME SofaVolumetricData
    VERSION ${SOFAVOLUMETRICDATA_VERSION}
    TARGETS ${PROJECT_NAME}
    INCLUDE_INSTALL_DIR "SofaVolumetricData"
    RELOCATABLE "plugins"
    )
