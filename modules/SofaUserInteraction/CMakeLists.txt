cmake_minimum_required(VERSION 3.1)
project(SofaUserInteraction)

message(STATUS "SofaUserInteraction:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
set(HEADER_FILES
    config.h
    initUserInteraction.h
    )

set(SOURCE_FILES
    initUserInteraction.cpp
    )

############################## COMPONENTS HERE ARE THE FULL-SET ####################################
if(SOFA_BUILD_FULLSETCOMPONENTS)
    list(APPEND HEADER_FILES
        # AddFramePerformer.h
        # AddFramePerformer.inl
        AttachBodyPerformer.h
        AttachBodyPerformer.inl
        ComponentMouseInteraction.h
        ComponentMouseInteraction.inl
        Controller.h
        InteractionPerformer.h
        MechanicalStateController.h
        MechanicalStateController.inl
        MouseInteractor.h
        MouseInteractor.inl
        Ray.h
        RayContact.h
        RayDiscreteIntersection.h
        RayDiscreteIntersection.inl
        RayModel.h
        RayNewProximityIntersection.h
        RayTraceDetection.h
        TopologicalChangeManager.h
        )

    list(APPEND SOURCE_FILES
        # AddFramePerformer.cpp
        AttachBodyPerformer.cpp
        ComponentMouseInteraction.cpp
        Controller.cpp
        InteractionPerformer.cpp
        MechanicalStateController.cpp
        MouseInteractor.cpp
        RayContact.cpp
        RayDiscreteIntersection.cpp
        RayModel.cpp
        RayNewProximityIntersection.cpp
        RayTraceDetection.cpp
        TopologicalChangeManager.cpp
        )
    message(STATUS "   With all maintained sofa components.")
else()
    message(STATUS "   With only a minimal set of components.")
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_DEPRECATEDCOMPONENTS)
    list(APPEND HEADER_FILES
        AddRecordedCameraPerformer.h
        ArticulatedHierarchyBVHController.h
        ArticulatedHierarchyController.h
        CuttingPoint.h
        DisabledContact.h
        EdgeSetController.h
        EdgeSetController.inl
        FixParticlePerformer.h
        FixParticlePerformer.inl
        GraspingManager.h
        InciseAlongPathPerformer.h
        InterpolationController.h
        InterpolationController.inl
        MechanicalStateControllerOmni.h
        MechanicalStateControllerOmni.inl
        NodeToggleController.h
        RemovePrimitivePerformer.h
        RemovePrimitivePerformer.inl
        SleepController.h
        StartNavigationPerformer.h
        SuturePointPerformer.h
        SuturePointPerformer.inl
        )
    list(APPEND SOURCE_FILES
        AddRecordedCameraPerformer.cpp
        ArticulatedHierarchyBVHController.cpp
        ArticulatedHierarchyController.cpp
        CuttingPoint.cpp
        DisabledContact.cpp
        EdgeSetController.cpp
        FixParticlePerformer.cpp
        GraspingManager.cpp
        InciseAlongPathPerformer.cpp
        InterpolationController.cpp
        MechanicalStateControllerOmni.cpp
        NodeToggleController.cpp
        RemovePrimitivePerformer.cpp
        SleepController.cpp
        StartNavigationPerformer.cpp
        SuturePointPerformer.cpp
        )
    message(STATUS "   With deprecated components.")
else()
    message(STATUS "   Without deprecated components.")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaGeneralMeshCollision SofaGeneralRigid SofaGeneralVisual SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent SofaBaseVisual)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_USER_INTERACTION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
