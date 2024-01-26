cmake_minimum_required(VERSION 3.12)
project(Sofa.Core LANGUAGES CXX)

set(SRC_ROOT "src/sofa/core")

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/AccumulationVecId.h
    ${SRC_ROOT}/AccumulationVecId.inl
    ${SRC_ROOT}/BaseMatrixAccumulatorComponent.h
    ${SRC_ROOT}/BaseLocalMappingMatrix.h
    ${SRC_ROOT}/BaseMapping.h
    ${SRC_ROOT}/BaseState.h
    ${SRC_ROOT}/BehaviorModel.h
    ${SRC_ROOT}/CategoryLibrary.h
    ${SRC_ROOT}/CollisionElement.h
    ${SRC_ROOT}/CollisionModel.h
    ${SRC_ROOT}/ComponentLibrary.h
    ${SRC_ROOT}/ComponentNameHelper.h
    ${SRC_ROOT}/ConstraintOrder.h
    ${SRC_ROOT}/ConstraintParams.h
    ${SRC_ROOT}/DataEngine.h
    ${SRC_ROOT}/DataTracker.h
    ${SRC_ROOT}/DataTrackerCallback.h
    ${SRC_ROOT}/DataTrackerFunctor.h
    ${SRC_ROOT}/DerivativeMatrix.h
    ${SRC_ROOT}/DevBaseMonitor.h
    ${SRC_ROOT}/ExecParams.h
    ${SRC_ROOT}/fwd.h
    ${SRC_ROOT}/init.h
    ${SRC_ROOT}/Mapping.h
    ${SRC_ROOT}/Mapping.inl
    ${SRC_ROOT}/MappingHelper.h
    ${SRC_ROOT}/MatrixAccumulator.h
    ${SRC_ROOT}/MechanicalParams.h
    ${SRC_ROOT}/MechanicalStatesMatrixAccumulators.h
    ${SRC_ROOT}/Multi2Mapping.h
    ${SRC_ROOT}/Multi2Mapping.inl
    ${SRC_ROOT}/MultiMapping.h
    ${SRC_ROOT}/MultiMapping.inl
    ${SRC_ROOT}/MultiVecId.h
    ${SRC_ROOT}/ObjectFactory.h
    ${SRC_ROOT}/PathResolver.h
    ${SRC_ROOT}/SofaLibrary.h
    ${SRC_ROOT}/sptr.h
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
    ${SRC_ROOT}/behavior/BaseLinearSolver.h
    ${SRC_ROOT}/behavior/BaseLocalForceFieldMatrix.h
    ${SRC_ROOT}/behavior/BaseLocalMassMatrix.h
    ${SRC_ROOT}/behavior/BaseMass.h
    ${SRC_ROOT}/behavior/BaseMatrixLinearSystem.h
    ${SRC_ROOT}/behavior/BaseMechanicalState.h
    ${SRC_ROOT}/behavior/BaseProjectiveConstraintSet.h
    ${SRC_ROOT}/behavior/BaseRotationFinder.h
    ${SRC_ROOT}/behavior/BaseVectorOperations.h
    ${SRC_ROOT}/behavior/BlockType.h
    ${SRC_ROOT}/behavior/BlocMatrixWriter.h
    ${SRC_ROOT}/behavior/Constraint.h
    ${SRC_ROOT}/behavior/Constraint.inl
    ${SRC_ROOT}/behavior/ConstraintCorrection.h
    ${SRC_ROOT}/behavior/ConstraintCorrection.inl
    ${SRC_ROOT}/behavior/ConstraintResolution.h
    ${SRC_ROOT}/behavior/ConstraintSolver.h
    ${SRC_ROOT}/behavior/DefaultMultiMatrixAccessor.h
    ${SRC_ROOT}/behavior/ForceField.h
    ${SRC_ROOT}/behavior/ForceField.inl
    ${SRC_ROOT}/behavior/LinearSolver.h
    ${SRC_ROOT}/behavior/MatrixAPICompatibility.h
    ${SRC_ROOT}/behavior/Mass.h
    ${SRC_ROOT}/behavior/Mass.inl
    ${SRC_ROOT}/behavior/MechanicalMatrix.h
    ${SRC_ROOT}/behavior/MechanicalState.h
    ${SRC_ROOT}/behavior/MechanicalState.inl
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
    ${SRC_ROOT}/behavior/PairStateAccessor.h
    ${SRC_ROOT}/behavior/ProjectiveConstraintSet.h
    ${SRC_ROOT}/behavior/ProjectiveConstraintSet.inl
    ${SRC_ROOT}/behavior/RotationFinder.h
    ${SRC_ROOT}/behavior/StateAccessor.h
    ${SRC_ROOT}/behavior/SingleMatrixAccessor.h
    ${SRC_ROOT}/behavior/SingleStateAccessor.h
    ${SRC_ROOT}/behavior/fwd.h
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
    ${SRC_ROOT}/loader/BaseLoader.h
    ${SRC_ROOT}/loader/ImageLoader.h
    ${SRC_ROOT}/loader/MeshLoader.h
    ${SRC_ROOT}/loader/SceneLoader.h
    ${SRC_ROOT}/loader/VoxelLoader.h
    ${SRC_ROOT}/logging/PerComponentLoggingMessageHandler.h
    ${SRC_ROOT}/logging/RichConsoleStyleMessageFormatter.h
    ${SRC_ROOT}/objectmodel/AbstractDataLink.h
    ${SRC_ROOT}/objectmodel/AspectPool.h
    ${SRC_ROOT}/objectmodel/Base.h
    ${SRC_ROOT}/objectmodel/BaseClass.h
    ${SRC_ROOT}/objectmodel/BaseClassNameHelper.h
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
    ${SRC_ROOT}/objectmodel/DataCallback.h
    ${SRC_ROOT}/objectmodel/DataContentValue.h
    ${SRC_ROOT}/objectmodel/DataFileName.h
    ${SRC_ROOT}/objectmodel/DataFileNameVector.h
    ${SRC_ROOT}/objectmodel/DataLink.h
    ${SRC_ROOT}/objectmodel/DeprecatedData.h
    ${SRC_ROOT}/objectmodel/RemovedData.h
    ${SRC_ROOT}/objectmodel/DetachNodeEvent.h
    ${SRC_ROOT}/objectmodel/Event.h
    ${SRC_ROOT}/objectmodel/GUIEvent.h
    ${SRC_ROOT}/objectmodel/HapticDeviceEvent.h
    ${SRC_ROOT}/objectmodel/IdleEvent.h
    ${SRC_ROOT}/objectmodel/JoystickEvent.h
    ${SRC_ROOT}/objectmodel/KeypressedEvent.h
    ${SRC_ROOT}/objectmodel/KeyreleasedEvent.h
    ${SRC_ROOT}/objectmodel/Link.h
    ${SRC_ROOT}/objectmodel/MouseEvent.h
    ${SRC_ROOT}/objectmodel/SPtr.h
    ${SRC_ROOT}/objectmodel/ScriptEvent.h
    ${SRC_ROOT}/objectmodel/TypeOfInsertion.h
    ${SRC_ROOT}/objectmodel/Tag.h
    ${SRC_ROOT}/objectmodel/TagSet.h
    ${SRC_ROOT}/objectmodel/vectorData.h
    ${SRC_ROOT}/objectmodel/vectorLinks.h
    ${SRC_ROOT}/topology/BaseMeshTopology.h
    ${SRC_ROOT}/topology/BaseTopology.h
    ${SRC_ROOT}/topology/BaseTopologyData.h
    ${SRC_ROOT}/topology/BaseTopologyObject.h
    ${SRC_ROOT}/topology/TopologicalMapping.h
    ${SRC_ROOT}/topology/Topology.h
    ${SRC_ROOT}/topology/TopologyChange.h
    ${SRC_ROOT}/topology/TopologyHandler.h
    ${SRC_ROOT}/topology/TopologyData.h
    ${SRC_ROOT}/topology/TopologyData.inl
    ${SRC_ROOT}/topology/TopologyDataHandler.h
    ${SRC_ROOT}/topology/TopologyDataHandler.inl
    ${SRC_ROOT}/topology/TopologySubsetData.h
    ${SRC_ROOT}/topology/TopologySubsetData.inl
    ${SRC_ROOT}/topology/TopologySubsetData.cpp
    ${SRC_ROOT}/topology/TopologySubsetIndices.h
    ${SRC_ROOT}/visual/Data[DisplayFlags].h
    ${SRC_ROOT}/visual/DisplayFlags.h
    ${SRC_ROOT}/visual/FlagTreeItem.h
    ${SRC_ROOT}/visual/Shader.h
    ${SRC_ROOT}/visual/Tristate.h
    ${SRC_ROOT}/visual/VisualLoop.h
    ${SRC_ROOT}/visual/VisualManager.h
    ${SRC_ROOT}/visual/VisualModel.h
    ${SRC_ROOT}/visual/VisualParams.h
    ${SRC_ROOT}/visual/VisualState.h
    ${SRC_ROOT}/visual/VisualState.inl
)

set(SOURCE_FILES
    ${SRC_ROOT}/AccumulationVecId.cpp
    ${SRC_ROOT}/BaseMatrixAccumulatorComponent.cpp
    ${SRC_ROOT}/BaseMapping.cpp
    ${SRC_ROOT}/BaseState.cpp
    ${SRC_ROOT}/BehaviorModel.cpp
    ${SRC_ROOT}/CategoryLibrary.cpp
    ${SRC_ROOT}/CollisionModel.cpp
    ${SRC_ROOT}/ComponentLibrary.cpp
    ${SRC_ROOT}/ComponentNameHelper.cpp
    ${SRC_ROOT}/ConstraintParams.cpp
    ${SRC_ROOT}/DataEngine.cpp
    ${SRC_ROOT}/DataTracker.cpp
    ${SRC_ROOT}/DataTrackerCallback.cpp
    ${SRC_ROOT}/DataTrackerFunctor.cpp
    ${SRC_ROOT}/ExecParams.cpp
    ${SRC_ROOT}/fwd.cpp
    ${SRC_ROOT}/init.cpp
    ${SRC_ROOT}/Mapping.cpp
    ${SRC_ROOT}/MatrixAccumulator.cpp
    ${SRC_ROOT}/MechanicalParams.cpp
    ${SRC_ROOT}/Multi2Mapping.cpp
    ${SRC_ROOT}/MultiMapping.cpp
    ${SRC_ROOT}/ObjectFactory.cpp
    ${SRC_ROOT}/PathResolver.cpp
    ${SRC_ROOT}/SofaLibrary.cpp
    ${SRC_ROOT}/State.cpp
    ${SRC_ROOT}/VecId.cpp
    ${SRC_ROOT}/behavior/BaseAnimationLoop.cpp
    ${SRC_ROOT}/behavior/BaseConstraint.cpp
    ${SRC_ROOT}/behavior/BaseConstraintCorrection.cpp
    ${SRC_ROOT}/behavior/BaseConstraintSet.cpp
    ${SRC_ROOT}/behavior/BaseForceField.cpp
    ${SRC_ROOT}/behavior/BaseInteractionForceField.cpp
    ${SRC_ROOT}/behavior/BaseLinearSolver.cpp
    ${SRC_ROOT}/behavior/BaseMass.cpp
    ${SRC_ROOT}/behavior/BaseMatrixLinearSystem.cpp
    ${SRC_ROOT}/behavior/BaseMechanicalState.cpp
    ${SRC_ROOT}/behavior/BaseProjectiveConstraintSet.cpp
    ${SRC_ROOT}/behavior/Constraint.cpp
    ${SRC_ROOT}/behavior/ConstraintCorrection.cpp
    ${SRC_ROOT}/behavior/ConstraintResolution.cpp
    ${SRC_ROOT}/behavior/ConstraintSolver.cpp
    ${SRC_ROOT}/behavior/DefaultMultiMatrixAccessor.cpp
    ${SRC_ROOT}/behavior/ForceField.cpp
    ${SRC_ROOT}/behavior/LinearSolver.cpp
    ${SRC_ROOT}/behavior/Mass.cpp
    ${SRC_ROOT}/behavior/MechanicalMatrix.cpp
    ${SRC_ROOT}/behavior/MechanicalState.cpp
    ${SRC_ROOT}/behavior/MixedInteractionConstraint.cpp
    ${SRC_ROOT}/behavior/MixedInteractionForceField.cpp
    ${SRC_ROOT}/behavior/MultiMatrixAccessor.cpp
    ${SRC_ROOT}/behavior/OdeSolver.cpp
    ${SRC_ROOT}/behavior/PairInteractionConstraint.cpp
    ${SRC_ROOT}/behavior/PairInteractionForceField.cpp
    ${SRC_ROOT}/behavior/PairInteractionProjectiveConstraintSet.cpp
    ${SRC_ROOT}/behavior/ProjectiveConstraintSet.cpp
    ${SRC_ROOT}/behavior/SingleMatrixAccessor.cpp
    ${SRC_ROOT}/behavior/fwd.cpp
    ${SRC_ROOT}/collision/BroadPhaseDetection.cpp
    ${SRC_ROOT}/collision/Contact.cpp
    ${SRC_ROOT}/collision/Intersection.cpp
    ${SRC_ROOT}/collision/NarrowPhaseDetection.cpp
    ${SRC_ROOT}/collision/Pipeline.cpp
    ${SRC_ROOT}/loader/BaseLoader.cpp
    ${SRC_ROOT}/loader/MeshLoader.cpp
    ${SRC_ROOT}/loader/SceneLoader.cpp
    ${SRC_ROOT}/loader/VoxelLoader.cpp
    ${SRC_ROOT}/logging/PerComponentLoggingMessageHandler.cpp
    ${SRC_ROOT}/logging/RichConsoleStyleMessageFormatter.cpp
    ${SRC_ROOT}/objectmodel/AbstractDataLink.cpp
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
    ${SRC_ROOT}/objectmodel/Context.cpp
    ${SRC_ROOT}/objectmodel/ContextObject.cpp
    ${SRC_ROOT}/objectmodel/DDGNode.cpp
    ${SRC_ROOT}/objectmodel/Data.cpp
    ${SRC_ROOT}/objectmodel/DataCallback.cpp
    ${SRC_ROOT}/objectmodel/DataContentValue.cpp
    ${SRC_ROOT}/objectmodel/DataFileName.cpp
    ${SRC_ROOT}/objectmodel/DataFileNameVector.cpp
    ${SRC_ROOT}/objectmodel/DataLink.cpp
    ${SRC_ROOT}/objectmodel/DetachNodeEvent.cpp
    ${SRC_ROOT}/objectmodel/DeprecatedData.cpp
    ${SRC_ROOT}/objectmodel/Event.cpp
    ${SRC_ROOT}/objectmodel/GUIEvent.cpp
    ${SRC_ROOT}/objectmodel/HapticDeviceEvent.cpp
    ${SRC_ROOT}/objectmodel/IdleEvent.cpp
    ${SRC_ROOT}/objectmodel/JoystickEvent.cpp
    ${SRC_ROOT}/objectmodel/KeypressedEvent.cpp
    ${SRC_ROOT}/objectmodel/KeyreleasedEvent.cpp
    ${SRC_ROOT}/objectmodel/MouseEvent.cpp
    ${SRC_ROOT}/objectmodel/ScriptEvent.cpp
    ${SRC_ROOT}/objectmodel/Tag.cpp
    ${SRC_ROOT}/objectmodel/TagSet.cpp
    ${SRC_ROOT}/topology/BaseMeshTopology.cpp
    ${SRC_ROOT}/topology/BaseTopology.cpp
    ${SRC_ROOT}/topology/BaseTopologyObject.cpp
    ${SRC_ROOT}/topology/TopologicalMapping.cpp
    ${SRC_ROOT}/topology/Topology.cpp
    ${SRC_ROOT}/topology/TopologyChange.cpp
    ${SRC_ROOT}/topology/TopologyHandler.cpp
    ${SRC_ROOT}/topology/TopologyData.cpp
    ${SRC_ROOT}/topology/TopologySubsetIndices.cpp
    ${SRC_ROOT}/visual/Data[DisplayFlags].cpp
    ${SRC_ROOT}/visual/DisplayFlags.cpp
    ${SRC_ROOT}/visual/FlagTreeItem.cpp
    ${SRC_ROOT}/visual/Shader.cpp
    ${SRC_ROOT}/visual/VisualLoop.cpp
    ${SRC_ROOT}/visual/VisualManager.cpp
    ${SRC_ROOT}/visual/VisualModel.cpp
    ${SRC_ROOT}/visual/VisualParams.cpp
    ${SRC_ROOT}/visual/VisualState.cpp
)

sofa_find_package(Sofa.Topology REQUIRED)
sofa_find_package(Sofa.Helper REQUIRED)
sofa_find_package(Sofa.DefaultType REQUIRED)

# to move to core ?
option(SOFA_CORE_ENABLE_CRSMULTIMATRIXACCESSOR "Use CompressedRowSparseMatrix to store global matrix (caution: it will lead to compile CRSMultiMatrixAcessor, which takes a considerable time to compile)" OFF)
mark_as_advanced(FORCE SOFA_CORE_ENABLE_CRSMULTIMATRIXACCESSOR)
if(SOFA_CORE_ENABLE_CRSMULTIMATRIXACCESSOR)
    list(APPEND HEADER_FILES ${SRC_ROOT}/behavior/CRSMultiMatrixAccessor.h)
    list(APPEND SOURCE_FILES ${SRC_ROOT}/behavior/CRSMultiMatrixAccessor.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Helper Sofa.Topology Sofa.DefaultType)

if(SOFA_BUILD_WITH_PCH_ENABLED)
    message("Adding precompiled header for Sofa.Core")
    target_precompile_headers(${PROJECT_NAME} PUBLIC ${SRC_ROOT}/objectmodel/BaseObject.h ${SRC_ROOT}/objectmodel/Data.h)
endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

# SOFA-NG: make everybody be able to use Sofa.Component.Compat
sofa_find_package(Sofa.Component.Compat QUIET)
if(Sofa.Component.Compat_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Compat)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES FOLDER Sofa.Framework) # IDE folder

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "${PROJECT_NAME}"
)

# Tests
# If SOFA_BUILD_TESTS exists and is OFF, then these tests will be auto-disabled
cmake_dependent_option(SOFA_CORE_BUILD_TESTS "Compile the automatic tests" ON "SOFA_BUILD_TESTS OR NOT DEFINED SOFA_BUILD_TESTS" OFF)
if(SOFA_CORE_BUILD_TESTS)
    add_subdirectory(test)
    add_subdirectory(simutest)
endif()
