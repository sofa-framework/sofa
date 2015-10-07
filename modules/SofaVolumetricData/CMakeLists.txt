cmake_minimum_required(VERSION 2.8.12)
project(SofaVolumetricData)

set(HEADER_FILES
    ImplicitSurfaceContainer.h
    ImplicitSurfaceMapping.h
    ImplicitSurfaceMapping.inl
    InterpolatedImplicitSurface.h
    InterpolatedImplicitSurface.inl
    config.h
    initVolumetricData.h
)

set(SOURCE_FILES
    ImplicitSurfaceContainer.cpp
    ImplicitSurfaceMapping.cpp
    InterpolatedImplicitSurface.cpp
    initVolumetricData.cpp
)

if(SOFA_HAVE_MINIFLOWVR)

    list(APPEND HEADER_FILES
        DistanceGrid.h
        DistanceGridCollisionModel.h
        DistanceGridForceField.h
        DistanceGridForceField.inl
        FFDDistanceGridDiscreteIntersection.h
        FFDDistanceGridDiscreteIntersection.inl
        RigidDistanceGridDiscreteIntersection.h
        RigidDistanceGridDiscreteIntersection.inl
    )

    list(APPEND SOURCE_FILES
        BarycentricPenalityContact_DistanceGrid.cpp
        DistanceGrid.cpp
        DistanceGridCollisionModel.cpp
        DistanceGridForceField.cpp
        FFDDistanceGridDiscreteIntersection.cpp
        RayDistanceGridContact.cpp
        RigidDistanceGridDiscreteIntersection.cpp
    )

endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision ${SOFAADVANCED_DEPENDENCY_LINK})
if(SOFA_HAVE_MINIFLOWVR)
    target_link_libraries(${PROJECT_NAME} PUBLIC SofaUserInteraction)
    target_link_libraries(${PROJECT_NAME} PUBLIC miniFlowVR)
endif()
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_VOLUMETRIC_DATA")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaAdvanced ${PROJECT_NAME} ${PROJECT_NAME})
