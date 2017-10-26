cmake_minimum_required(VERSION 3.1)
project(SofaUserInteraction)

message(STATUS "SofaUserInteraction:")
################################ COMPONENTS HERE ARE THE NG-SET ####################################
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

############################## COMPONENTS HERE ARE THE LIGHT-SET ###################################
if(SOFA_BUILD_COMPONENTSET_LIGHT)
    list(APPEND HEADER_FILES

        )
    list(APPEND SOURCE_FILES

        )
endif()

############################## COMPONENTS HERE ARE THE STANDARD-SET ################################
if(SOFA_BUILD_COMPONENTSET_STANDARD)
    list(APPEND HEADER_FILES
        # AddFramePerformer.h
        # AddFramePerformer.inl
        AttachBodyPerformer.h
        AttachBodyPerformer.inl
        ComponentMouseInteraction.h
        ComponentMouseInteraction.inl
        Controller.h
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
        )

    list(APPEND SOURCE_FILES
        # AddFramePerformer.cpp
        AttachBodyPerformer.cpp
        ComponentMouseInteraction.cpp
        Controller.cpp
        MechanicalStateController.cpp
        RayContact.cpp
        RayDiscreteIntersection.cpp
        RayModel.cpp
        RayNewProximityIntersection.cpp
        RayTraceDetection.cpp
        SleepController.cpp
        TopologicalChangeManager.cpp
        )
endif()

############################### COMPONENTS HERE ARE DEPRECATED ####################################
if(SOFA_BUILD_COMPONENTSET_FULL)
    list(APPEND HEADER_FILES
        AddRecordedCameraPerformer.h
        ArticulatedHierarchyBVHController.h
        ArticulatedHierarchyController.h
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
        StartNavigationPerformer.h
        SuturePointPerformer.h
        SuturePointPerformer.inl
        )
    list(APPEND SOURCE_FILES
        AddRecordedCameraPerformer.cpp
        ArticulatedHierarchyBVHController.cpp
        ArticulatedHierarchyController.cpp
        DisabledContact.cpp
        EdgeSetController.cpp
        FixParticlePerformer.cpp
        GraspingManager.cpp
        InciseAlongPathPerformer.cpp
        InterpolationController.cpp
        MechanicalStateControllerOmni.cpp
        NodeToggleController.cpp
        RemovePrimitivePerformer.cpp
        StartNavigationPerformer.cpp
        SuturePointPerformer.cpp
        )
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaMeshCollision SofaGeneralMeshCollision SofaGeneralRigid SofaGeneralVisual SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent SofaBaseVisual)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_USER_INTERACTION")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGeneral ${PROJECT_NAME} ${PROJECT_NAME})
