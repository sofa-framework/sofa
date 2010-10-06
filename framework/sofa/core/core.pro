# Subdir relative project main directory: ./framework/sofa/core
# Target is a library:  sofacore$$LIBSUFFIX

SOFA_DIR = ../../..
TEMPLATE = lib
TARGET = sofacore

include($${SOFA_DIR}/sofa.cfg)

CONFIG += $$CONFIGLIBRARIES

!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
    CONFIG += dll
}

LIBS += -lsofahelper$$LIBSUFFIX -lsofadefaulttype$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS

DEFINES += SOFA_BUILD_CORE

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_DIR/modules
INCLUDEPATH -= $$SOFA_DIR/applications

HEADERS += \
          core.h \
          VecId.h \
          ExecParams.h \
          MechanicalParams.h \
          BaseMapping.h \
          BehaviorModel.h \
          CollisionElement.h \
          CollisionModel.h \
          DataEngine.h \
          behavior/BaseController.h \
          behavior/BaseConstraint.h \
          behavior/BaseConstraintSet.h \
          behavior/BaseConstraintCorrection.h \
          behavior/BaseForceField.h \
          behavior/BaseLMConstraint.h \
          behavior/BaseMass.h \
          behavior/BaseMechanicalMapping.h \
          behavior/BaseMechanicalState.h \ 
          behavior/BaseProjectiveConstraintSet.h \
          behavior/Constraint.h \
          behavior/Constraint.inl \
          behavior/ConstraintSolver.h \
          behavior/ForceField.h \
          behavior/ForceField.inl \
          behavior/InteractionForceField.h \
          behavior/InteractionProjectiveConstraintSet.h \
          behavior/InteractionConstraint.h \
          behavior/LinearSolver.h \
          behavior/LMConstraint.h \
          behavior/LMConstraint.inl \
          behavior/PairInteractionForceField.h \
          behavior/PairInteractionForceField.inl \
          behavior/MixedInteractionForceField.h \
          behavior/MixedInteractionForceField.inl \
          behavior/PairInteractionConstraint.h \
          behavior/PairInteractionConstraint.inl \
          behavior/PairInteractionProjectiveConstraintSet.h \
          behavior/PairInteractionProjectiveConstraintSet.inl \
          behavior/MixedInteractionConstraint.h \
          behavior/MixedInteractionConstraint.inl \
          behavior/MappedModel.h \
          behavior/Mass.h \
          behavior/Mass.inl \
          behavior/MasterSolver.h \
          behavior/MechanicalMapping.h \
          behavior/MechanicalMapping.inl \
		  behavior/MechanicalMultiMapping.h \
		  behavior/MechanicalMultiMapping.inl \
		  behavior/MechanicalMulti2Mapping.h \
		  behavior/MechanicalMulti2Mapping.inl \
          behavior/MechanicalState.h \
          behavior/MultiVector.h \
          behavior/MultiMatrix.h \
          behavior/MultiMatrixAccessor.h \
          behavior/ProjectiveConstraintSet.h \
          behavior/ProjectiveConstraintSet.inl \
          behavior/OdeSolver.h \
          behavior/State.h \
          collision/BroadPhaseDetection.h \
          collision/CollisionAlgorithm.h \
          collision/CollisionGroupManager.h \
          collision/Contact.h \
          collision/ContactManager.h \
          collision/Detection.h \
          collision/DetectionOutput.h \
          collision/Intersection.h \
          collision/Intersection.inl \
          collision/NarrowPhaseDetection.h \
          collision/Pipeline.h \
          topology/BaseMeshTopology.h \
          topology/BaseTopology.h \
          topology/BaseTopologyObject.h \
          topology/TopologicalMapping.h \
          topology/Topology.h \
          loader/BaseLoader.h \
          loader/ImageLoader.h \
          loader/Material.h \
          loader/MeshLoader.h \
          loader/PrimitiveGroup.h \
          Mapping.h \
          Mapping.inl \
          MultiMapping.h \
          MultiMapping.inl \
          Multi2Mapping.h \
          Multi2Mapping.inl \
          objectmodel/Base.h \
          objectmodel/BaseClass.h \
          objectmodel/BaseContext.h \
          objectmodel/BaseNode.h \
          objectmodel/BaseObject.h \
          objectmodel/BaseObjectDescription.h \
          objectmodel/ClassInfo.h \
          objectmodel/ConfigurationSetting.h \
          objectmodel/Context.h \
          objectmodel/ContextObject.h \
          objectmodel/Data.h \
          objectmodel/DataFileName.h \
		  objectmodel/DDGNode.h \
          objectmodel/DetachNodeEvent.h \
          objectmodel/Event.h \
          objectmodel/BaseData.h \
	  objectmodel/JoystickEvent.h \
          objectmodel/KeypressedEvent.h \
	  objectmodel/KeyreleasedEvent.h \
	  objectmodel/MouseEvent.h \
	  objectmodel/OmniEvent.h \
	  objectmodel/GLInitializedEvent.h \
          objectmodel/Tag.h \
          VisualModel.h \
          VisualManager.h \
          ObjectFactory.h \
          SofaLibrary.h \
          CategoryLibrary.h \
          ComponentLibrary.h 

SOURCES +=  \
          objectmodel/Base.cpp \
          objectmodel/BaseClass.cpp \
          objectmodel/BaseData.cpp \
          objectmodel/BaseContext.cpp \
          objectmodel/BaseObject.cpp \
          objectmodel/BaseObjectDescription.cpp \
          objectmodel/ClassInfo.cpp \
          objectmodel/ConfigurationSetting.cpp \
          objectmodel/Context.cpp \
          objectmodel/Data.cpp \
	  objectmodel/DDGNode.cpp \
          objectmodel/DetachNodeEvent.cpp \
          objectmodel/Event.cpp \
          objectmodel/DataFileName.cpp \
	  objectmodel/JoystickEvent.cpp \
          objectmodel/KeypressedEvent.cpp \
          objectmodel/KeyreleasedEvent.cpp \
	  objectmodel/MouseEvent.cpp \
  	  objectmodel/OmniEvent.cpp \
          objectmodel/Tag.cpp \
          ObjectFactory.cpp \
          SofaLibrary.cpp \
          CategoryLibrary.cpp \
          ComponentLibrary.cpp \
          Mapping.cpp \	   
          MultiMapping.cpp \
          Multi2Mapping.cpp \
          CollisionModel.cpp \
          DataEngine.cpp \
          behavior/LinearSolver.cpp \
          behavior/MasterSolver.cpp \
          behavior/MultiMatrix.cpp \
          behavior/MultiMatrixAccessor.cpp \
          behavior/OdeSolver.cpp \
          behavior/ConstraintSolver.cpp \
          collision/Contact.cpp \
          collision/Intersection.cpp \
          collision/Pipeline.cpp \
          topology/BaseMeshTopology.cpp \
          topology/BaseTopology.cpp \
          behavior/BaseConstraint.cpp \
          behavior/BaseForceField.cpp \   
          behavior/BaseLMConstraint.cpp \
          behavior/ForceField.cpp \
          behavior/LMConstraint.cpp \
          behavior/Mass.cpp \
          behavior/Constraint.cpp \
          behavior/MechanicalMapping.cpp \
		  behavior/MechanicalMultiMapping.cpp \
		  behavior/MechanicalMulti2Mapping.cpp \
          behavior/PairInteractionForceField.cpp \
          behavior/MixedInteractionForceField.cpp \
          behavior/PairInteractionConstraint.cpp \
          behavior/PairInteractionProjectiveConstraintSet.cpp \
          behavior/MixedInteractionConstraint.cpp \
          behavior/ProjectiveConstraintSet.cpp \
          loader/MeshLoader.cpp 

contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV

HEADERS += \
	 fem/FEMRecipe.h                 \
	 fem/QuadratureFormular.h        \
	 fem/FENodes.h			         \
 	 fem/FENodes.inl		         \
	 fem/BaseMaterial.h              \
	 fem/BaseStrainTensor.h          \
	 fem/BaseFiniteElement.h         \
	 fem/FiniteElement.h             \
	 fem/DofContainer.h              \
  	 fem/PolynomialInterpolation.h   \
 	 fem/PolynomialQuadratureInterpolation.h   \
 	 fem/PolynomialQuadratureInterpolation.inl \   
 	 fem/StrainTensor.h              \    
 	 fem/Material.h                 
 
SOURCES +=  \
	 fem/FENodes.cpp		         \
	 fem/BaseMaterial.cpp            \
	 fem/BaseFiniteElement.cpp       \
	 fem/FiniteElement.cpp           \
	 fem/DofContainer.cpp            \
 	 fem/BaseStrainTensor.cpp        \
 	 fem/PolynomialQuadratureInterpolation.cpp
 
} # END SOFA_DEV


contains(DEFINES,SOFA_SMP){
HEADERS +=  \
					CallContext.h \
          objectmodel/BaseObjectTasks.h \
          ParallelCollisionModel.h \
          behavior/ParallelMultiVector.h \
          collision/ParallelPipeline.h \
          collision/ParallelNarrowPhaseDetection.h

SOURCES +=  \
					CallContext.cpp \
          collision/ParallelPipeline.cpp\
          objectmodel/BaseObjectTasks.cpp 
}
