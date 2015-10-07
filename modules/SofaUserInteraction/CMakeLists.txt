cmake_minimum_required(VERSION 2.8.12)
project(SofaUserInteraction)

set(HEADER_FILES
    # AddFramePerformer.h
    # AddFramePerformer.inl
    AddRecordedCameraPerformer.h
    ArticulatedHierarchyBVHController.h
    ArticulatedHierarchyController.h
    AttachBodyPerformer.h
    AttachBodyPerformer.inl
    ComponentMouseInteraction.h
    ComponentMouseInteraction.inl
    Controller.h
    CuttingPoint.h
    DisabledContact.h
    EdgeSetController.h
    EdgeSetController.inl
    FixParticlePerformer.h
    FixParticlePerformer.inl
    GraspingManager.h
    InciseAlongPathPerformer.h
    InteractionPerformer.h
    InterpolationController.h
    MechanicalStateController.h
    MechanicalStateController.inl
    MechanicalStateControllerOmni.h
    MechanicalStateControllerOmni.inl
    MouseInteractor.h
    MouseInteractor.inl
    NodeToggleController.h
    Ray.h
    RayContact.h
    RayDiscreteIntersection.h
    RayDiscreteIntersection.inl
    RayModel.h
    RayNewProximityIntersection.h
    RayTraceDetection.h
    RemovePrimitivePerformer.h
    RemovePrimitivePerformer.inl
    SleepController.h
    StartNavigationPerformer.h
    SuturePointPerformer.h
    SuturePointPerformer.inl
    TopologicalChangeManager.h
    config.h
    initUserInteraction.h
)

set(SOURCE_FILES
    # AddFramePerformer.cpp
    AddRecordedCameraPerformer.cpp
    ArticulatedHierarchyBVHController.cpp
    ArticulatedHierarchyController.cpp
    AttachBodyPerformer.cpp
    ComponentMouseInteraction.cpp
    Controller.cpp
    CuttingPoint.cpp
    DisabledContact.cpp
    EdgeSetController.cpp
    FixParticlePerformer.cpp
    GraspingManager.cpp
    InciseAlongPathPerformer.cpp
    InteractionPerformer.cpp
    InterpolationController.cpp
    InterpolationController.inl
    MechanicalStateController.cpp
    MechanicalStateControllerOmni.cpp
    MouseInteractor.cpp
    NodeToggleController.cpp
    RayContact.cpp
    RayDiscreteIntersection.cpp
    RayModel.cpp
    RayNewProximityIntersection.cpp
    RayTraceDetection.cpp
    RemovePrimitivePerformer.cpp
    SleepController.cpp
    StartNavigationPerformer.cpp
    SuturePointPerformer.cpp
    TopologicalChangeManager.cpp
    initUserInteraction.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent SofaBaseVisual)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_USER_INTERACTION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
