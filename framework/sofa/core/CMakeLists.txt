cmake_minimum_required(VERSION 2.8.8)
project(SofaCore)


set(SOFACORE_HEADERS
	VecId.h
#	Plugin.h
#	PluginManager.h
	ConstraintParams.h
	ExecParams.h
	MechanicalParams.h
	MultiVecId.h
	BaseMapping.h
	BaseState.h
	State.h
	State.inl
	BehaviorModel.h
	CollisionElement.h
	CollisionModel.h
	DataEngine.h
	behavior/BaseAnimationLoop.h
	behavior/BaseController.h
	behavior/BaseConstraint.h
	behavior/BaseConstraintSet.h
	behavior/BaseConstraintCorrection.h
	behavior/BaseForceField.h
	behavior/BaseInteractionForceField.h
	behavior/BaseInteractionProjectiveConstraintSet.h
	behavior/BaseInteractionConstraint.h
	behavior/BaseLMConstraint.h
	behavior/BaseMass.h
	behavior/BaseMechanicalState.h
	behavior/BaseProjectiveConstraintSet.h
	behavior/BaseVectorOperations.h
	behavior/BaseRotationFinder.h
	behavior/RotationMatrix.h
	behavior/Constraint.h
	behavior/Constraint.inl
	behavior/ConstraintCorrection.h
	behavior/ConstraintCorrection.inl
	behavior/ConstraintSolver.h
	behavior/ForceField.h
	behavior/ForceField.inl
	behavior/LinearSolver.h
	behavior/LMConstraint.h
	behavior/LMConstraint.inl
	behavior/PairInteractionForceField.h
	behavior/PairInteractionForceField.inl
	behavior/MixedInteractionForceField.h
	behavior/MixedInteractionForceField.inl
	behavior/PairInteractionConstraint.h
	behavior/PairInteractionConstraint.inl
	behavior/PairInteractionProjectiveConstraintSet.h
	behavior/PairInteractionProjectiveConstraintSet.inl
	behavior/MixedInteractionConstraint.h
	behavior/MixedInteractionConstraint.inl
	behavior/Mass.h
	behavior/Mass.inl
	behavior/MechanicalState.h
	behavior/MultiVec.h
	behavior/MultiMatrix.h
	behavior/MultiMatrixAccessor.h
	behavior/ProjectiveConstraintSet.h
	behavior/ProjectiveConstraintSet.inl
	behavior/OdeSolver.h
	behavior/RotationFinder.h
	collision/BroadPhaseDetection.h
	collision/CollisionAlgorithm.h
	collision/CollisionGroupManager.h
	collision/Contact.h
	collision/ContactManager.h
	collision/Detection.h
	collision/DetectionOutput.h
	collision/Intersection.h
	collision/Intersection.inl
	collision/NarrowPhaseDetection.h
	collision/Pipeline.h
	collision/IntersectorFactory.h
	topology/BaseMeshTopology.h
	topology/BaseTopology.h
	topology/BaseTopologyObject.h
	topology/TopologicalMapping.h
	topology/TopologyChange.h
	topology/BaseTopologyData.h
	topology/BaseTopologyEngine.h
	topology/TopologyHandler.h
	topology/TopologyElementHandler.h
	topology/Topology.h
	loader/BaseLoader.h
	loader/ImageLoader.h
	loader/Material.h
	loader/MeshLoader.h
	loader/PrimitiveGroup.h
	loader/VoxelLoader.h
	loader/SceneLoader.h
	Mapping.h
	Mapping.inl
	MultiMapping.h
	MultiMapping.inl
	Multi2Mapping.h
	Multi2Mapping.inl
	objectmodel/AspectPool.h
	objectmodel/SPtr.h
	objectmodel/Base.h
	objectmodel/BaseClass.h
	objectmodel/BaseContext.h
	objectmodel/BaseNode.h
	objectmodel/BaseObject.h
	objectmodel/BaseObjectDescription.h
	objectmodel/ClassInfo.h
	objectmodel/ConfigurationSetting.h
	objectmodel/Context.h
	objectmodel/ContextObject.h
	objectmodel/Data.h
	objectmodel/DataFileName.h
	objectmodel/DDGNode.h
	objectmodel/DetachNodeEvent.h
	objectmodel/Event.h
	objectmodel/BaseData.h
	objectmodel/BaseLink.h
	objectmodel/Link.h
	objectmodel/HapticDeviceEvent.h
	objectmodel/JoystickEvent.h
	objectmodel/KeypressedEvent.h
	objectmodel/KeyreleasedEvent.h
	objectmodel/MouseEvent.h
	objectmodel/Tag.h
	visual/DisplayFlags.h
	visual/VisualParams.h
	visual/VisualLoop.h
	visual/VisualModel.h
	visual/VisualManager.h
	visual/DrawTool.h
	visual/Shader.h
	ObjectFactory.h
	SofaLibrary.h
	CategoryLibrary.h
	ComponentLibrary.h
	objectmodel/GUIEvent.h
	)

set(SOFACORE_SOURCES
	objectmodel/AspectPool.cpp
	objectmodel/Base.cpp
	objectmodel/BaseClass.cpp
	objectmodel/BaseData.cpp
	objectmodel/BaseLink.cpp
	objectmodel/BaseContext.cpp
	objectmodel/BaseNode.cpp
	objectmodel/BaseObject.cpp
	objectmodel/BaseObjectDescription.cpp
	objectmodel/ClassInfo.cpp
	objectmodel/ConfigurationSetting.cpp
	objectmodel/Context.cpp
	objectmodel/Data.cpp
	objectmodel/DDGNode.cpp
	objectmodel/DetachNodeEvent.cpp
	objectmodel/Event.cpp
	objectmodel/DataFileName.cpp
	objectmodel/JoystickEvent.cpp
	objectmodel/KeypressedEvent.cpp
	objectmodel/KeyreleasedEvent.cpp
	objectmodel/MouseEvent.cpp
	objectmodel/Tag.cpp
	ObjectFactory.cpp
	ExecParams.cpp
	MechanicalParams.cpp
#	Plugin.cpp
#	PluginManager.cpp
	ConstraintParams.cpp
	SofaLibrary.cpp
	CategoryLibrary.cpp
	ComponentLibrary.cpp
	BaseMapping.cpp
	Mapping.cpp
	MultiMapping.cpp
	Multi2Mapping.cpp
	CollisionModel.cpp
	DataEngine.cpp
	State.cpp
	behavior/BaseAnimationLoop.cpp
	behavior/LinearSolver.cpp
	behavior/MultiMatrix.cpp
	behavior/MultiMatrixAccessor.cpp
	behavior/OdeSolver.cpp
	behavior/ConstraintSolver.cpp
	collision/Contact.cpp
	collision/Intersection.cpp
	collision/Pipeline.cpp
	topology/BaseMeshTopology.cpp
	topology/BaseTopology.cpp
	topology/TopologyChange.cpp
	topology/TopologyHandler.cpp
	topology/TopologyElementHandler.cpp
	behavior/BaseConstraint.cpp
	behavior/BaseForceField.cpp
	behavior/BaseLMConstraint.cpp
	behavior/BaseMechanicalState.cpp
	behavior/MechanicalState.cpp
	behavior/ForceField.cpp
	behavior/LMConstraint.cpp
	behavior/Mass.cpp
	behavior/Constraint.cpp
	behavior/ConstraintCorrection.cpp
	behavior/PairInteractionForceField.cpp
	behavior/MixedInteractionForceField.cpp
	behavior/PairInteractionConstraint.cpp
	behavior/PairInteractionProjectiveConstraintSet.cpp
	behavior/MixedInteractionConstraint.cpp
	behavior/ProjectiveConstraintSet.cpp
	loader/MeshLoader.cpp
	loader/VoxelLoader.cpp
	loader/SceneLoader.cpp
	visual/DisplayFlags.cpp
	visual/VisualParams.cpp
	objectmodel/HapticDeviceEvent.cpp
	objectmodel/GUIEvent.cpp
	)

if(NOT SOFA-MISC_NO_OPENGL)
	if(SOFA_HAVE_GLUT)
		list(APPEND SOFACORE_HEADERS "visual/DrawToolGL.h")
		list(APPEND SOFACORE_SOURCES "visual/DrawToolGL.cpp")
	endif()
endif()  
  
if(Boost_thread_FOUND)
	list(APPEND SOFACORE_HEADERS "objectmodel/AspectPool.h")
	list(APPEND SOFACORE_SOURCES "objectmodel/AspectPool.cpp")
endif()

if(SOFA-MISC_SMP)
	list(APPEND SOFACORE_HEADERS "CallContext.h")
	list(APPEND SOFACORE_HEADERS "objectmodel/BaseObjectTasks.h")
	list(APPEND SOFACORE_HEADERS "ParallelCollisionModel.h")
	list(APPEND SOFACORE_HEADERS "behavior/ParallelMultiVec.h")
	list(APPEND SOFACORE_HEADERS "collision/ParallelPipeline.h")
	list(APPEND SOFACORE_HEADERS "collision/ParallelNarrowPhaseDetection.h")
	
	list(APPEND SOFACORE_SOURCES "CallContext.cpp")
	list(APPEND SOFACORE_SOURCES "collision/ParallelPipeline.cpp")
	list(APPEND SOFACORE_SOURCES "objectmodel/BaseObjectTasks.cpp")
endif()

if(SOFA-MISC_SUPPORT_MOVING_FRAMES)
	list(APPEND SOFACORE_HEADERS "behavior/InertiaForce.h")
endif()

if(PS3)
	list(APPEND SOFACORE_HEADERS "ps3/cxa_demangle.h")
	list(APPEND SOFACORE_HEADERS "ps3/cxxabi.h")
	list(APPEND SOFACORE_SOURCES "ps3/cxa_demangle.cpp")
endif()

add_library(${PROJECT_NAME} SHARED ${SOFACORE_HEADERS} ${SOFACORE_SOURCES})
target_link_libraries( ${PROJECT_NAME} PUBLIC SofaDefaultType)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../>") 

set( SOFACORE_COMPILER_FLAGS "-DSOFA_BUILD_CORE")
# make the name of debug libraries end in _d.
set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAFRAMEWORK_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFACORE_COMPILER_FLAGS}"  )

install(TARGETS ${PROJECT_NAME}
    COMPONENT SofaFramework_libraries
    EXPORT  SofaFrameworkTargets
    RUNTIME DESTINATION "${SOFAFRAMEWORK_INSTALL_BIN_DIR}"
    LIBRARY DESTINATION "${SOFAFRAMEWORK_INSTALL_LIB_DIR}"
    ARCHIVE DESTINATION "${SOFAFRAMEWORK_INSTALL_LIB_DIR}"
    )
 
install(DIRECTORY ${SofaCore_SOURCE_DIR}/ 
        DESTINATION "${SOFAFRAMEWORK_INSTALL_INCLUDE_DIR}/sofa/core"
        COMPONENT SofaFramework_headers
        FILES_MATCHING 
            PATTERN "*.h"
            PATTERN "*.inl"
            PATTERN "*.hxx"
            PATTERN "*.hpp"
        )


