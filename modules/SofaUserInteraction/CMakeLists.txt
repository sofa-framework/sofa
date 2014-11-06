cmake_minimum_required(VERSION 2.8)
project(SofaUserInteraction)

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
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries( ${PROJECT_NAME} SofaMeshCollision SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent SofaBaseVisual )
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGENERAL_DEPENDENCY_INCLUDE_DIR}>")

set( SOFAUSER_INTERACTION_COMPILER_FLAGS "-DSOFA_BUILD_USER_INTERACTION")

set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGENERAL_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAUSER_INTERACTION_COMPILER_FLAGS}"  )
set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" )

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaUserInteraction_libraries
        EXPORT    SofaGeneralTargets
        RUNTIME DESTINATION "${SOFAGENERAL_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFAGENERAL_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFAGENERAL_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}"

)


