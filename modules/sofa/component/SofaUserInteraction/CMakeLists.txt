include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    ../initUserInteraction.h 
    ../collision/RayTraceDetection.h 
    ../collision/RayContact.h 
    # ../collision/DisabledContact.h 
     ../collision/ComponentMouseInteraction.h
     ../collision/ComponentMouseInteraction.inl
    ../collision/MouseInteractor.h 
    ../collision/MouseInteractor.inl 
    ../collision/AttachBodyPerformer.h 
    ../collision/AttachBodyPerformer.inl 
     ../collision/FixParticlePerformer.h 
     ../collision/FixParticlePerformer.inl 
     ../collision/InteractionPerformer.h
     ../collision/SuturePointPerformer.h 
     ../collision/SuturePointPerformer.inl 
    # ../collision/GraspingManager.h 
    # ../controller/ArticulatedHierarchyController.h 
    # ../controller/ArticulatedHierarchyBVHController.h 
     ../controller/Controller.h
    # ../controller/EdgeSetController.h 
    # ../controller/EdgeSetController.inl 
    # ../controller/MechanicalStateController.h 
    # ../controller/MechanicalStateController.inl 
    ../collision/Ray.h 
    ../collision/RayModel.h 
    # ../collision/RayDiscreteIntersection.h 
    # ../collision/RayDiscreteIntersection.inl 
    # ../collision/RayNewProximityIntersection.h 
     ../collision/RemovePrimitivePerformer.h 
     ../collision/RemovePrimitivePerformer.inl 
     ../collision/InciseAlongPathPerformer.h 
     ../collision/TopologicalChangeManager.h
     ../collision/CuttingPoint.h 
# #    ../collision/AddFramePerformer.h
#    ../collision/AddFramePerformer.inl
     ../collision/AddRecordedCameraPerformer.h 
     ../collision/StartNavigationPerformer.h
#     ../controller/MechanicalStateControllerOmni.h 
# 	../controller/MechanicalStateControllerOmni.inl 
#     ../controller/NodeToggleController.h
#     ../controller/InterpolationController.h
# 	../controller/SleepController.h
    )
    
set(SOURCE_FILES

    ../initUserInteraction.cpp 
#    ../collision/RayTraceDetection.cpp
    ../collision/RayContact.cpp 
    # ../collision/DisabledContact.cpp 
     ../collision/ComponentMouseInteraction.cpp
    ../collision/MouseInteractor.cpp 
    ../collision/AttachBodyPerformer.cpp 
     ../collision/FixParticlePerformer.cpp 
     ../collision/InteractionPerformer.cpp
     ../collision/SuturePointPerformer.cpp 
    # ../collision/GraspingManager.cpp 
    # ../controller/ArticulatedHierarchyController.cpp 
    # ../controller/ArticulatedHierarchyBVHController.cpp 
     ../controller/Controller.cpp
    # ../controller/EdgeSetController.cpp 
    # ../controller/MechanicalStateController.cpp 
    ../collision/RayModel.cpp 
    # ../collision/RayDiscreteIntersection.cpp 
    # ../collision/RayNewProximityIntersection.cpp 
     ../collision/RemovePrimitivePerformer.cpp 
     ../collision/InciseAlongPathPerformer.cpp 
     ../collision/TopologicalChangeManager.cpp
     ../collision/CuttingPoint.cpp 
    # ../collision/AddFramePerformer.cpp
     ../collision/AddRecordedCameraPerformer.cpp
     ../collision/StartNavigationPerformer.cpp
 #    ../controller/MechanicalStateControllerOmni.cpp 
 #    ../controller/NodeToggleController.cpp
 #    ../controller/InterpolationController.cpp
 #    ../controller/InterpolationController.inl
	# ../controller/SleepController.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_USER_INTERACTION" )
AddLinkerDependencies(SofaMeshCollision SofaTopologyMapping SofaDeformable SofaBoundaryCondition SofaGraphComponent SofaBaseVisual )
    
include(${SOFA_CMAKE_DIR}/postProject.cmake)
