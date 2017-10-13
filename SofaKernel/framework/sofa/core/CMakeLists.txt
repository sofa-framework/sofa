cmake_minimum_required(VERSION 3.1)
project(SofaCore)

set(HEADER_FILES
    BaseMapping.h
    BaseState.h
    BehaviorModel.h
    CategoryLibrary.h
    CollisionElement.h
    CollisionModel.h
    ComponentLibrary.h
    ConstraintParams.h
    DataEngine.h
    DataTracker.h
    ExecParams.h
    Mapping.h
    Mapping.inl
    MechanicalParams.h
    Multi2Mapping.h
    Multi2Mapping.inl
    MultiMapping.h
    MultiMapping.inl
    MultiVecId.h
    ObjectFactory.h
    SofaLibrary.h
    State.h
    State.inl
    VecId.h
    behavior/BaseAnimationLoop.h
    behavior/BaseConstraint.h
    behavior/BaseConstraintCorrection.h
    behavior/BaseConstraintSet.h
    behavior/BaseController.h
    behavior/BaseForceField.h
    behavior/BaseInteractionConstraint.h
    behavior/BaseInteractionForceField.h
    behavior/BaseInteractionProjectiveConstraintSet.h
    behavior/BaseLMConstraint.h
    behavior/BaseMass.h
    behavior/BaseMechanicalState.h
    behavior/BaseProjectiveConstraintSet.h
    behavior/BaseRotationFinder.h
    behavior/BaseVectorOperations.h
    behavior/Constraint.h
    behavior/Constraint.inl
    behavior/ConstraintCorrection.h
    behavior/ConstraintCorrection.inl
    behavior/ConstraintSolver.h
    behavior/ForceField.h
    behavior/ForceField.inl
    behavior/LMConstraint.h
    behavior/LMConstraint.inl
    behavior/LinearSolver.h
    behavior/Mass.h
    behavior/Mass.inl
    behavior/MechanicalState.h
    behavior/MixedInteractionConstraint.h
    behavior/MixedInteractionConstraint.inl
    behavior/MixedInteractionForceField.h
    behavior/MixedInteractionForceField.inl
    behavior/MultiMatrix.h
    behavior/MultiMatrixAccessor.h
    behavior/MultiVec.h
    behavior/OdeSolver.h
    behavior/PairInteractionConstraint.h
    behavior/PairInteractionConstraint.inl
    behavior/PairInteractionForceField.h
    behavior/PairInteractionForceField.inl
    behavior/PairInteractionProjectiveConstraintSet.h
    behavior/PairInteractionProjectiveConstraintSet.inl
    behavior/ProjectiveConstraintSet.h
    behavior/ProjectiveConstraintSet.inl
    behavior/RotationMatrix.h
    collision/BroadPhaseDetection.h
    collision/CollisionAlgorithm.h
    collision/CollisionGroupManager.h
    collision/Contact.h
    collision/ContactManager.h
    collision/Detection.h
    collision/DetectionOutput.h
    collision/Intersection.h
    collision/Intersection.inl
    collision/IntersectorFactory.h
    collision/NarrowPhaseDetection.h
    collision/Pipeline.h
    core.h
    init.h
    loader/BaseLoader.h
    loader/ImageLoader.h
    loader/Material.h
    loader/MeshLoader.h
    loader/PrimitiveGroup.h
    loader/SceneLoader.h
    loader/VoxelLoader.h
    objectmodel/AspectPool.h
    objectmodel/Base.h
    objectmodel/BaseClass.h
    objectmodel/BaseContext.h
    objectmodel/BaseData.h
    objectmodel/BaseLink.h
    objectmodel/BaseNode.h
    objectmodel/BaseObject.h
    objectmodel/BaseObjectDescription.h
    objectmodel/ClassInfo.h
    objectmodel/ConfigurationSetting.h
    objectmodel/Context.h
    objectmodel/ContextObject.h
    objectmodel/DDGNode.h
    objectmodel/Data.h
    objectmodel/DataFileName.h
    objectmodel/DetachNodeEvent.h
    objectmodel/Event.h
    objectmodel/GUIEvent.h
    objectmodel/HapticDeviceEvent.h
    objectmodel/JoystickEvent.h
    objectmodel/KeypressedEvent.h
    objectmodel/KeyreleasedEvent.h
    objectmodel/IdleEvent.h
    objectmodel/Link.h
    objectmodel/MouseEvent.h
    objectmodel/SPtr.h
    objectmodel/Tag.h
    topology/BaseMeshTopology.h
    topology/BaseTopology.h
    topology/BaseTopologyData.h
    topology/BaseTopologyEngine.h
    topology/BaseTopologyObject.h
    topology/TopologicalMapping.h
    topology/Topology.h
    topology/TopologyChange.h
    topology/TopologyElementHandler.h
    topology/TopologyHandler.h
    visual/DisplayFlags.h
    visual/DrawTool.h
    visual/Shader.h
    visual/VisualLoop.h
    visual/VisualManager.h
    visual/VisualModel.h
    visual/VisualParams.h
    logging/RichConsoleStyleMessageFormatter.h
    logging/PerComponentLoggingMessageHandler.h
)

set(SOURCE_FILES
    BaseMapping.cpp
    BaseState.cpp
    BehaviorModel.cpp
    CategoryLibrary.cpp
    CollisionModel.cpp
    ComponentLibrary.cpp
    ConstraintParams.cpp
    DataEngine.cpp
    DataTracker.cpp
    ExecParams.cpp
    Mapping.cpp
    MechanicalParams.cpp
    Multi2Mapping.cpp
    MultiMapping.cpp
    ObjectFactory.cpp
    # Plugin.cpp
    # PluginManager.cpp
    SofaLibrary.cpp
    State.cpp
    behavior/BaseAnimationLoop.cpp
    behavior/BaseConstraintSet.cpp
    behavior/BaseForceField.cpp
    behavior/BaseInteractionForceField.cpp
    behavior/BaseLMConstraint.cpp
    behavior/BaseMass.cpp
    behavior/BaseMechanicalState.cpp
    behavior/BaseProjectiveConstraintSet.cpp
    behavior/Constraint.cpp
    behavior/ConstraintCorrection.cpp
    behavior/ConstraintSolver.cpp
    behavior/ForceField.cpp
    behavior/LMConstraint.cpp
    behavior/LinearSolver.cpp
    behavior/Mass.cpp
    behavior/MechanicalState.cpp
    behavior/MixedInteractionConstraint.cpp
    behavior/MixedInteractionForceField.cpp
    behavior/MultiMatrix.cpp
    behavior/MultiMatrixAccessor.cpp
    behavior/OdeSolver.cpp
    behavior/PairInteractionConstraint.cpp
    behavior/PairInteractionForceField.cpp
    behavior/PairInteractionProjectiveConstraintSet.cpp
    behavior/ProjectiveConstraintSet.cpp
    collision/Contact.cpp
    collision/Intersection.cpp
    collision/Pipeline.cpp
    init.cpp
    loader/BaseLoader.cpp
    loader/MeshLoader.cpp
    loader/SceneLoader.cpp
    loader/VoxelLoader.cpp
    objectmodel/AspectPool.cpp
    objectmodel/Base.cpp
    objectmodel/BaseClass.cpp
    objectmodel/BaseContext.cpp
    objectmodel/BaseData.cpp
    objectmodel/BaseLink.cpp
    objectmodel/BaseNode.cpp
    objectmodel/BaseObject.cpp
    objectmodel/BaseObjectDescription.cpp
    objectmodel/ClassInfo.cpp
    objectmodel/ConfigurationSetting.cpp
    objectmodel/ContextObject.cpp
    objectmodel/Context.cpp
    objectmodel/DDGNode.cpp
    objectmodel/Data.cpp
    objectmodel/DataFileName.cpp
    objectmodel/DetachNodeEvent.cpp
    objectmodel/Event.cpp
    objectmodel/GUIEvent.cpp
    objectmodel/HapticDeviceEvent.cpp
    objectmodel/JoystickEvent.cpp
    objectmodel/KeypressedEvent.cpp
    objectmodel/KeyreleasedEvent.cpp
    objectmodel/MouseEvent.cpp
    objectmodel/IdleEvent.cpp
    objectmodel/Tag.cpp
    topology/BaseMeshTopology.cpp
    topology/BaseTopology.cpp
    topology/BaseTopologyObject.cpp
    topology/Topology.cpp
    topology/TopologyChange.cpp
    topology/TopologyElementHandler.cpp
    topology/TopologyHandler.cpp
    visual/Shader.cpp
    visual/DisplayFlags.cpp
    visual/VisualLoop.cpp
    visual/VisualManager.cpp
    visual/VisualModel.cpp
    visual/VisualParams.cpp
    logging/RichConsoleStyleMessageFormatter.cpp
    logging/PerComponentLoggingMessageHandler.cpp
)

if(NOT SOFA_NO_OPENGL)
    list(APPEND HEADER_FILES visual/DrawToolGL.h)
    list(APPEND SOURCE_FILES visual/DrawToolGL.cpp)
endif()

if(Boost_thread_FOUND)
    list(APPEND HEADER_FILES objectmodel/AspectPool.h)
    list(APPEND SOURCE_FILES objectmodel/AspectPool.cpp)
endif()



# # TODO?
# if(SOFA-MISC_SUPPORT_MOVING_FRAMES)
#     list(APPEND HEADER_FILES behavior/InertiaForce.h)
# endif()

if(PS3)
    list(APPEND HEADER_FILES ps3/cxa_demangle.h)
    list(APPEND HEADER_FILES ps3/cxxabi.h)
    list(APPEND SOURCE_FILES ps3/cxa_demangle.cpp)
endif()


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaHelper SofaDefaultType)
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>")
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_CORE")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION "${SOFAFRAMEWORK_VERSION}")

sofa_install_targets(SofaFramework SofaCore "")

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/" DESTINATION "include/sofa/core"
        FILES_MATCHING PATTERN "*.h" PATTERN "*.inl")
