cmake_minimum_required(VERSION 3.1)
project(SofaUserInteraction)

set(HEADER_FILES
    InteractionPerformer.h
    MouseInteractor.h
    MouseInteractor.inl
    config.h
    initUserInteraction.h
    )

set(SOURCE_FILES
    InteractionPerformer.cpp
    MouseInteractor.cpp
    initUserInteraction.cpp
    )

list(APPEND HEADER_FILES
    # AddFramePerformer.h
    # AddFramePerformer.inl
    AddRecordedCameraPerformer.h
    AttachBodyPerformer.h
    AttachBodyPerformer.inl
    ComponentMouseInteraction.h
    ComponentMouseInteraction.inl
    Controller.h
    FixParticlePerformer.h
    FixParticlePerformer.inl
    InciseAlongPathPerformer.h
    MechanicalStateController.h
    MechanicalStateController.inl
    Ray.h
    RayContact.h
    RayDiscreteIntersection.h
    RayDiscreteIntersection.inl
    RayModel.h
    RayNewProximityIntersection.h
    RayTraceDetection.h
    SleepController.h
    TopologicalChangeManager.h
    RemovePrimitivePerformer.h
    RemovePrimitivePerformer.inl
    StartNavigationPerformer.h
    SuturePointPerformer.h
    SuturePointPerformer.inl
    )

list(APPEND SOURCE_FILES
    # AddFramePerformer.cpp
    AddRecordedCameraPerformer.cpp
    AttachBodyPerformer.cpp
    ComponentMouseInteraction.cpp
    Controller.cpp
    FixParticlePerformer.cpp
    InciseAlongPathPerformer.cpp
    MechanicalStateController.cpp
    RayContact.cpp
    RayDiscreteIntersection.cpp
    RayModel.cpp
    RayNewProximityIntersection.cpp
    RayTraceDetection.cpp
    SleepController.cpp
    TopologicalChangeManager.cpp
    RemovePrimitivePerformer.cpp
    StartNavigationPerformer.cpp
    SuturePointPerformer.cpp
    )

if(SOFA_WITH_DEPRECATED_COMPONENTS)
    list(APPEND HEADER_FILES
        ArticulatedHierarchyBVHController.h
        ArticulatedHierarchyController.h
        DisabledContact.h
        EdgeSetController.h
        EdgeSetController.inl
        GraspingManager.h
        InterpolationController.h
        InterpolationController.inl
        MechanicalStateControllerOmni.h
        MechanicalStateControllerOmni.inl
        NodeToggleController.h
        )
    list(APPEND SOURCE_FILES
        ArticulatedHierarchyBVHController.cpp
        ArticulatedHierarchyController.cpp
        DisabledContact.cpp
        EdgeSetController.cpp
        GraspingManager.cpp
        InterpolationController.cpp
        MechanicalStateControllerOmni.cpp
        NodeToggleController.cpp
        )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaGeneralMeshCollision SofaGeneralRigid SofaGeneralVisual SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent SofaBaseVisual)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_USER_INTERACTION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
