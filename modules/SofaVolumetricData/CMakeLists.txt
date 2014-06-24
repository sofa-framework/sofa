include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initVolumetricData.h
    ImplicitSurfaceContainer.h
    InterpolatedImplicitSurface.h
    InterpolatedImplicitSurface.inl
    DistanceGridForceField.h
    DistanceGridForceField.inl
    ImplicitSurfaceMapping.h
    ImplicitSurfaceMapping.inl
    DistanceGrid.h
    DistanceGridCollisionModel.h
    RigidDistanceGridDiscreteIntersection.h
    RigidDistanceGridDiscreteIntersection.inl
    FFDDistanceGridDiscreteIntersection.h
    FFDDistanceGridDiscreteIntersection.inl

    )

set(SOURCE_FILES

    initVolumetricData.cpp
    ImplicitSurfaceContainer.cpp
    InterpolatedImplicitSurface.cpp
    DistanceGridForceField.cpp
    ImplicitSurfaceMapping.cpp
    DistanceGrid.cpp
    DistanceGridCollisionModel.cpp
    RayDistanceGridContact.cpp
    RigidDistanceGridDiscreteIntersection.cpp
    FFDDistanceGridDiscreteIntersection.cpp
    BarycentricPenalityContact_DistanceGrid.cpp

    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_VOLUMETRIC_DATA" )
AddLinkerDependencies(SofaBaseCollision SofaMeshCollision SofaUserInteraction FlowVR)

include(${SOFA_CMAKE_DIR}/postProject.cmake)
