include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    initUserInteraction.h
    RayTraceDetection.h
    RayContact.h
    DisabledContact.h
    ComponentMouseInteraction.h
    ComponentMouseInteraction.inl
    MouseInteractor.h
    MouseInteractor.inl
    AttachBodyPerformer.h
    AttachBodyPerformer.inl
    FixParticlePerformer.h
    FixParticlePerformer.inl
    InteractionPerformer.h
    SuturePointPerformer.h
    SuturePointPerformer.inl
    GraspingManager.h
    ArticulatedHierarchyController.h
    ArticulatedHierarchyBVHController.h
    Controller.h
    EdgeSetController.h
    EdgeSetController.inl
    MechanicalStateController.h
    MechanicalStateController.inl
    Ray.h
    RayModel.h
    RayDiscreteIntersection.h
    RayDiscreteIntersection.inl
    RayNewProximityIntersection.h
    RemovePrimitivePerformer.h
    RemovePrimitivePerformer.inl
    InciseAlongPathPerformer.h
    TopologicalChangeManager.h
    CuttingPoint.h
#    AddFramePerformer.h
#    AddFramePerformer.inl
    AddRecordedCameraPerformer.h
    StartNavigationPerformer.h
    MechanicalStateControllerOmni.h
	MechanicalStateControllerOmni.inl
    NodeToggleController.h
    InterpolationController.h
    SleepController.h
    )

set(SOURCE_FILES

    initUserInteraction.cpp
    RayTraceDetection.cpp
    RayContact.cpp
    DisabledContact.cpp
    ComponentMouseInteraction.cpp
    MouseInteractor.cpp
    AttachBodyPerformer.cpp
    FixParticlePerformer.cpp
    InteractionPerformer.cpp
    SuturePointPerformer.cpp
    GraspingManager.cpp
    ArticulatedHierarchyController.cpp
    ArticulatedHierarchyBVHController.cpp
    Controller.cpp
    EdgeSetController.cpp
    MechanicalStateController.cpp
    RayModel.cpp
    RayDiscreteIntersection.cpp
    RayNewProximityIntersection.cpp
    RemovePrimitivePerformer.cpp
    InciseAlongPathPerformer.cpp
    TopologicalChangeManager.cpp
    CuttingPoint.cpp
#    AddFramePerformer.cpp
    AddRecordedCameraPerformer.cpp
    StartNavigationPerformer.cpp
    MechanicalStateControllerOmni.cpp
    NodeToggleController.cpp
    InterpolationController.cpp
    InterpolationController.inl
    SleepController.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_USER_INTERACTION" )
AddLinkerDependencies(SofaMeshCollision SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent SofaBaseVisual )

include(${SOFA_CMAKE_DIR}/postProject.cmake)
