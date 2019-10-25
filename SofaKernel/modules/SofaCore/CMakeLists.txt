cmake_minimum_required(VERSION 3.1)
project(SofaCore)

set(SRC_ROOT "src/sofa/core")

set(HEADER_FILES
    ${SRC_ROOT}/BaseMapping.h
    ${SRC_ROOT}/BaseState.h
    ${SRC_ROOT}/BehaviorModel.h
    ${SRC_ROOT}/CategoryLibrary.h
    ${SRC_ROOT}/CollisionElement.h
    ${SRC_ROOT}/CollisionModel.h
    ${SRC_ROOT}/ComponentLibrary.h
    ${SRC_ROOT}/ConstraintParams.h
    ${SRC_ROOT}/DataEngine.h
    ${SRC_ROOT}/DataTracker.h
    ${SRC_ROOT}/ExecParams.h
    ${SRC_ROOT}/Mapping.h
    ${SRC_ROOT}/Mapping.inl
    ${SRC_ROOT}/MechanicalParams.h
    ${SRC_ROOT}/Multi2Mapping.h
    ${SRC_ROOT}/Multi2Mapping.inl
    ${SRC_ROOT}/MultiMapping.h
    ${SRC_ROOT}/MultiMapping.inl
    ${SRC_ROOT}/MultiVecId.h
    ${SRC_ROOT}/ObjectFactory.h
    ${SRC_ROOT}/SofaLibrary.h
    ${SRC_ROOT}/State.h
    ${SRC_ROOT}/State.inl
    ${SRC_ROOT}/VecId.h
    ${SRC_ROOT}/behavior/BaseAnimationLoop.h
    ${SRC_ROOT}/behavior/BaseConstraint.h
    ${SRC_ROOT}/behavior/BaseConstraintCorrection.h
    ${SRC_ROOT}/behavior/BaseConstraintSet.h
    ${SRC_ROOT}/behavior/BaseController.h
    ${SRC_ROOT}/behavior/BaseForceField.h
    ${SRC_ROOT}/behavior/BaseInteractionConstraint.h
    ${SRC_ROOT}/behavior/BaseInteractionForceField.h
    ${SRC_ROOT}/behavior/BaseInteractionProjectiveConstraintSet.h
    ${SRC_ROOT}/behavior/BaseLMConstraint.h
    ${SRC_ROOT}/behavior/BaseMass.h
    ${SRC_ROOT}/behavior/BaseMechanicalState.h
    ${SRC_ROOT}/behavior/BaseProjectiveConstraintSet.h
    ${SRC_ROOT}/behavior/BaseRotationFinder.h
    ${SRC_ROOT}/behavior/BaseVectorOperations.h
    ${SRC_ROOT}/behavior/Constraint.h
    ${SRC_ROOT}/behavior/Constraint.inl
    ${SRC_ROOT}/behavior/ConstraintCorrection.h
    ${SRC_ROOT}/behavior/ConstraintCorrection.inl
    ${SRC_ROOT}/behavior/ConstraintSolver.h
    ${SRC_ROOT}/behavior/ForceField.h
    ${SRC_ROOT}/behavior/ForceField.inl
    ${SRC_ROOT}/behavior/LMConstraint.h
    ${SRC_ROOT}/behavior/LMConstraint.inl
    ${SRC_ROOT}/behavior/LinearSolver.h
    ${SRC_ROOT}/behavior/Mass.h
    ${SRC_ROOT}/behavior/Mass.inl
    ${SRC_ROOT}/behavior/MechanicalState.h
    ${SRC_ROOT}/behavior/MixedInteractionConstraint.h
    ${SRC_ROOT}/behavior/MixedInteractionConstraint.inl
    ${SRC_ROOT}/behavior/MixedInteractionForceField.h
    ${SRC_ROOT}/behavior/MixedInteractionForceField.inl
    ${SRC_ROOT}/behavior/MultiMatrix.h
    ${SRC_ROOT}/behavior/MultiMatrixAccessor.h
    ${SRC_ROOT}/behavior/MultiVec.h
    ${SRC_ROOT}/behavior/OdeSolver.h
    ${SRC_ROOT}/behavior/PairInteractionConstraint.h
    ${SRC_ROOT}/behavior/PairInteractionConstraint.inl
    ${SRC_ROOT}/behavior/PairInteractionForceField.h
    ${SRC_ROOT}/behavior/PairInteractionForceField.inl
    ${SRC_ROOT}/behavior/PairInteractionProjectiveConstraintSet.h
    ${SRC_ROOT}/behavior/PairInteractionProjectiveConstraintSet.inl
    ${SRC_ROOT}/behavior/ProjectiveConstraintSet.h
    ${SRC_ROOT}/behavior/ProjectiveConstraintSet.inl
    ${SRC_ROOT}/behavior/RotationMatrix.h
    ${SRC_ROOT}/collision/BroadPhaseDetection.h
    ${SRC_ROOT}/collision/CollisionAlgorithm.h
    ${SRC_ROOT}/collision/CollisionGroupManager.h
    ${SRC_ROOT}/collision/Contact.h
    ${SRC_ROOT}/collision/ContactManager.h
    ${SRC_ROOT}/collision/Detection.h
    ${SRC_ROOT}/collision/DetectionOutput.h
    ${SRC_ROOT}/collision/Intersection.h
    ${SRC_ROOT}/collision/Intersection.inl
    ${SRC_ROOT}/collision/IntersectorFactory.h
    ${SRC_ROOT}/collision/NarrowPhaseDetection.h
    ${SRC_ROOT}/collision/Pipeline.h
    ${SRC_ROOT}/core.h
    ${SRC_ROOT}/init.h
    ${SRC_ROOT}/loader/BaseLoader.h
    ${SRC_ROOT}/loader/ImageLoader.h
    ${SRC_ROOT}/loader/Material.h
    ${SRC_ROOT}/loader/MeshLoader.h
    ${SRC_ROOT}/loader/PrimitiveGroup.h
    ${SRC_ROOT}/loader/SceneLoader.h
    ${SRC_ROOT}/loader/VoxelLoader.h
    ${SRC_ROOT}/objectmodel/AspectPool.h
    ${SRC_ROOT}/objectmodel/Base.h
    ${SRC_ROOT}/objectmodel/BaseClass.h
    ${SRC_ROOT}/objectmodel/BaseContext.h
    ${SRC_ROOT}/objectmodel/BaseData.h
    ${SRC_ROOT}/objectmodel/BaseLink.h
    ${SRC_ROOT}/objectmodel/BaseNode.h
    ${SRC_ROOT}/objectmodel/BaseObject.h
    ${SRC_ROOT}/objectmodel/BaseObjectDescription.h
    ${SRC_ROOT}/objectmodel/ClassInfo.h
    ${SRC_ROOT}/objectmodel/ComponentState.h
    ${SRC_ROOT}/objectmodel/ConfigurationSetting.h
    ${SRC_ROOT}/objectmodel/Context.h
    ${SRC_ROOT}/objectmodel/ContextObject.h
    ${SRC_ROOT}/objectmodel/DDGNode.h
    ${SRC_ROOT}/objectmodel/Data.h
    ${SRC_ROOT}/objectmodel/DataFileName.h
    ${SRC_ROOT}/objectmodel/DetachNodeEvent.h
    ${SRC_ROOT}/objectmodel/Event.h
    ${SRC_ROOT}/objectmodel/GUIEvent.h
    ${SRC_ROOT}/objectmodel/HapticDeviceEvent.h
    ${SRC_ROOT}/objectmodel/JoystickEvent.h
    ${SRC_ROOT}/objectmodel/KeypressedEvent.h
    ${SRC_ROOT}/objectmodel/KeyreleasedEvent.h
    ${SRC_ROOT}/objectmodel/IdleEvent.h
    ${SRC_ROOT}/objectmodel/Link.h
    ${SRC_ROOT}/objectmodel/MouseEvent.h
    ${SRC_ROOT}/objectmodel/ScriptEvent.h
    ${SRC_ROOT}/objectmodel/SPtr.h
    ${SRC_ROOT}/objectmodel/Tag.h
    ${SRC_ROOT}/objectmodel/DataCallback.h
    ${SRC_ROOT}/topology/BaseMeshTopology.h
    ${SRC_ROOT}/topology/BaseTopology.h
    ${SRC_ROOT}/topology/BaseTopologyData.h
    ${SRC_ROOT}/topology/BaseTopologyEngine.h
    ${SRC_ROOT}/topology/BaseTopologyObject.h
    ${SRC_ROOT}/topology/TopologicalMapping.h
    ${SRC_ROOT}/topology/Topology.h
    ${SRC_ROOT}/topology/TopologyChange.h
    ${SRC_ROOT}/topology/TopologyElementHandler.h
    ${SRC_ROOT}/topology/TopologyHandler.h
    ${SRC_ROOT}/visual/DisplayFlags.h
    ${SRC_ROOT}/visual/DrawTool.h
    ${SRC_ROOT}/visual/Shader.h
    ${SRC_ROOT}/visual/VisualLoop.h
    ${SRC_ROOT}/visual/VisualManager.h
    ${SRC_ROOT}/visual/VisualModel.h
    ${SRC_ROOT}/visual/VisualParams.h
    ${SRC_ROOT}/logging/RichConsoleStyleMessageFormatter.h
    ${SRC_ROOT}/logging/PerComponentLoggingMessageHandler.h
)

set(SOURCE_FILES
    ${SRC_ROOT}/BaseMapping.cpp
    ${SRC_ROOT}/BaseState.cpp
    ${SRC_ROOT}/BehaviorModel.cpp
    ${SRC_ROOT}/CategoryLibrary.cpp
    ${SRC_ROOT}/CollisionModel.cpp
    ${SRC_ROOT}/ComponentLibrary.cpp
    ${SRC_ROOT}/ConstraintParams.cpp
    ${SRC_ROOT}/DataEngine.cpp
    ${SRC_ROOT}/DataTracker.cpp
    ${SRC_ROOT}/ExecParams.cpp
    ${SRC_ROOT}/Mapping.cpp
    ${SRC_ROOT}/MechanicalParams.cpp
    ${SRC_ROOT}/Multi2Mapping.cpp
    ${SRC_ROOT}/MultiMapping.cpp
    ${SRC_ROOT}/ObjectFactory.cpp
    # ${SRC_ROOT}/Plugin.cpp
    # ${SRC_ROOT}/PluginManager.cpp
    ${SRC_ROOT}/SofaLibrary.cpp
    ${SRC_ROOT}/State.cpp
    ${SRC_ROOT}/behavior/BaseAnimationLoop.cpp
    ${SRC_ROOT}/behavior/BaseConstraint.cpp
    ${SRC_ROOT}/behavior/BaseConstraintCorrection.cpp
    ${SRC_ROOT}/behavior/BaseConstraintSet.cpp
    ${SRC_ROOT}/behavior/BaseForceField.cpp
    ${SRC_ROOT}/behavior/BaseInteractionForceField.cpp
    ${SRC_ROOT}/behavior/BaseLMConstraint.cpp
    ${SRC_ROOT}/behavior/BaseMass.cpp
    ${SRC_ROOT}/behavior/BaseMechanicalState.cpp
    ${SRC_ROOT}/behavior/BaseProjectiveConstraintSet.cpp
    ${SRC_ROOT}/behavior/Constraint.cpp
    ${SRC_ROOT}/behavior/ConstraintCorrection.cpp
    ${SRC_ROOT}/behavior/ConstraintSolver.cpp
    ${SRC_ROOT}/behavior/ForceField.cpp
    ${SRC_ROOT}/behavior/LMConstraint.cpp
    ${SRC_ROOT}/behavior/LinearSolver.cpp
    ${SRC_ROOT}/behavior/Mass.cpp
    ${SRC_ROOT}/behavior/MechanicalState.cpp
    ${SRC_ROOT}/behavior/MixedInteractionConstraint.cpp
    ${SRC_ROOT}/behavior/MixedInteractionForceField.cpp
    ${SRC_ROOT}/behavior/MultiMatrix.cpp
    ${SRC_ROOT}/behavior/MultiMatrixAccessor.cpp
    ${SRC_ROOT}/behavior/OdeSolver.cpp
    ${SRC_ROOT}/behavior/PairInteractionConstraint.cpp
    ${SRC_ROOT}/behavior/PairInteractionForceField.cpp
    ${SRC_ROOT}/behavior/PairInteractionProjectiveConstraintSet.cpp
    ${SRC_ROOT}/behavior/ProjectiveConstraintSet.cpp
    ${SRC_ROOT}/collision/Contact.cpp
    ${SRC_ROOT}/collision/Intersection.cpp
    ${SRC_ROOT}/collision/Pipeline.cpp
    ${SRC_ROOT}/init.cpp
    ${SRC_ROOT}/loader/BaseLoader.cpp
    ${SRC_ROOT}/loader/MeshLoader.cpp
    ${SRC_ROOT}/loader/SceneLoader.cpp
    ${SRC_ROOT}/loader/VoxelLoader.cpp
    ${SRC_ROOT}/objectmodel/AspectPool.cpp
    ${SRC_ROOT}/objectmodel/Base.cpp
    ${SRC_ROOT}/objectmodel/BaseClass.cpp
    ${SRC_ROOT}/objectmodel/BaseContext.cpp
    ${SRC_ROOT}/objectmodel/BaseData.cpp
    ${SRC_ROOT}/objectmodel/BaseLink.cpp
    ${SRC_ROOT}/objectmodel/BaseNode.cpp
    ${SRC_ROOT}/objectmodel/BaseObject.cpp
    ${SRC_ROOT}/objectmodel/BaseObjectDescription.cpp
    ${SRC_ROOT}/objectmodel/ClassInfo.cpp
    ${SRC_ROOT}/objectmodel/ComponentState.cpp
    ${SRC_ROOT}/objectmodel/ConfigurationSetting.cpp
    ${SRC_ROOT}/objectmodel/ContextObject.cpp
    ${SRC_ROOT}/objectmodel/Context.cpp
    ${SRC_ROOT}/objectmodel/DDGNode.cpp
    ${SRC_ROOT}/objectmodel/Data.cpp
    ${SRC_ROOT}/objectmodel/DataCallback.cpp
    ${SRC_ROOT}/objectmodel/DataFileName.cpp
    ${SRC_ROOT}/objectmodel/DetachNodeEvent.cpp
    ${SRC_ROOT}/objectmodel/Event.cpp
    ${SRC_ROOT}/objectmodel/GUIEvent.cpp
    ${SRC_ROOT}/objectmodel/HapticDeviceEvent.cpp
    ${SRC_ROOT}/objectmodel/JoystickEvent.cpp
    ${SRC_ROOT}/objectmodel/KeypressedEvent.cpp
    ${SRC_ROOT}/objectmodel/KeyreleasedEvent.cpp
    ${SRC_ROOT}/objectmodel/MouseEvent.cpp
    ${SRC_ROOT}/objectmodel/ScriptEvent.cpp
    ${SRC_ROOT}/objectmodel/IdleEvent.cpp
    ${SRC_ROOT}/objectmodel/Tag.cpp
    ${SRC_ROOT}/topology/BaseMeshTopology.cpp
    ${SRC_ROOT}/topology/BaseTopology.cpp
    ${SRC_ROOT}/topology/BaseTopologyObject.cpp
    ${SRC_ROOT}/topology/Topology.cpp
    ${SRC_ROOT}/topology/TopologyChange.cpp
    ${SRC_ROOT}/topology/TopologyElementHandler.cpp
    ${SRC_ROOT}/topology/TopologyHandler.cpp
    ${SRC_ROOT}/visual/Shader.cpp
    ${SRC_ROOT}/visual/DisplayFlags.cpp
    ${SRC_ROOT}/visual/VisualLoop.cpp
    ${SRC_ROOT}/visual/VisualManager.cpp
    ${SRC_ROOT}/visual/VisualModel.cpp
    ${SRC_ROOT}/visual/VisualParams.cpp
    ${SRC_ROOT}/logging/RichConsoleStyleMessageFormatter.cpp
    ${SRC_ROOT}/logging/PerComponentLoggingMessageHandler.cpp
)

if(NOT SOFA_NO_OPENGL)
    list(APPEND HEADER_FILES ${SRC_ROOT}/visual/DrawToolGL.h)
    list(APPEND SOURCE_FILES ${SRC_ROOT}/visual/DrawToolGL.cpp)
endif()



# # TODO?
# if(SOFA-MISC_SUPPORT_MOVING_FRAMES)
#     list(APPEND HEADER_FILES behavior/InertiaForce.h)
# endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDefaultType)
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION "${SOFAFRAMEWORK_VERSION}")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_CORE")

sofa_install_targets(SofaFramework ${PROJECT_NAME} "SofaFramework/sofa/core" ${SRC_ROOT})

if(SOFA_BUILD_TESTS)
    add_subdirectory(SofaCore_test)
endif()
