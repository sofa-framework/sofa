cmake_minimum_required(VERSION 3.12)
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

find_package(SofaGeneralVisual REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaGeneralMeshCollision SofaGeneralRigid SofaGeneralVisual SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent SofaBaseVisual)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaGeneral
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    )
