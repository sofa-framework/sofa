cmake_minimum_required(VERSION 3.1)
project(SofaVolumetricData)

set(HEADER_FILES
    ImplicitSurfaceContainer.h
    ImplicitSurfaceMapping.h
    ImplicitSurfaceMapping.inl
    InterpolatedImplicitSurface.h
    InterpolatedImplicitSurface.inl
    config.h
    initVolumetricData.h
    DistanceGrid.h
    DistanceGrid.h
    DistanceGridCollisionModel.h
    DistanceGridForceField.h
    DistanceGridForceField.inl
    FFDDistanceGridDiscreteIntersection.h
    FFDDistanceGridDiscreteIntersection.inl
    RigidDistanceGridDiscreteIntersection.h
    RigidDistanceGridDiscreteIntersection.inl
)

set(SOURCE_FILES
    ImplicitSurfaceContainer.cpp
    ImplicitSurfaceMapping.cpp
    InterpolatedImplicitSurface.cpp
    initVolumetricData.cpp
    DistanceGrid.cpp
    BarycentricPenalityContact_DistanceGrid.cpp
    DistanceGridCollisionModel.cpp
    DistanceGridForceField.cpp
    FFDDistanceGridDiscreteIntersection.cpp
    RayDistanceGridContact.cpp
    RigidDistanceGridDiscreteIntersection.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision ${SOFAADVANCED_DEPENDENCY_LINK})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaUserInteraction)

if(SOFA_HAVE_MINIFLOWVR)
    target_link_libraries(${PROJECT_NAME} PUBLIC miniFlowVR)
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_VOLUMETRIC_DATA -DSOFA_HAVE_MINIFLOWVR")
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_VOLUMETRIC_DATA")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaAdvanced ${PROJECT_NAME} ${PROJECT_NAME})
