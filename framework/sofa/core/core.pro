# File generate by kdevelop's qmake manager. 
# ------------------------------------------- 
# Subdir relative project main directory: ./framework/sofa/core
# Target is a library:  sofacore$$LIBSUFFIX

SOFA_DIR = ../../..
TEMPLATE = lib
TARGET = sofacore

include($${SOFA_DIR}/sofa.cfg)

CONFIG += $$CONFIGLIBRARIES

CONFIG -= staticlib
CONFIG += dll

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
          componentmodel/behavior/BaseController.h \
          componentmodel/behavior/BaseConstraint.h \
          componentmodel/behavior/BaseConstraintCorrection.h \
          componentmodel/behavior/BaseForceField.h \
          componentmodel/behavior/BaseLMConstraint.h \
          componentmodel/behavior/BaseMass.h \
          componentmodel/behavior/BaseMechanicalMapping.h \
          componentmodel/behavior/BaseMechanicalState.h \
          componentmodel/behavior/Constraint.h \
          componentmodel/behavior/Constraint.inl \
          componentmodel/behavior/ConstraintSolver.h \
          componentmodel/behavior/ForceField.h \
          componentmodel/behavior/ForceField.inl \
          componentmodel/behavior/InteractionForceField.h \
          componentmodel/behavior/InteractionConstraint.h \
          componentmodel/behavior/LinearSolver.h \
          componentmodel/behavior/LMConstraint.h \
          componentmodel/behavior/LMConstraint.inl \
          componentmodel/behavior/PairInteractionForceField.h \
          componentmodel/behavior/PairInteractionForceField.inl \
	  componentmodel/behavior/MixedInteractionForceField.h \
	  componentmodel/behavior/MixedInteractionForceField.inl \
          componentmodel/behavior/PairInteractionConstraint.h \
          componentmodel/behavior/PairInteractionConstraint.inl \
	  componentmodel/behavior/MixedInteractionConstraint.h \
	  componentmodel/behavior/MixedInteractionConstraint.inl \
	  componentmodel/behavior/MappedModel.h \
          componentmodel/behavior/Mass.h \
          componentmodel/behavior/Mass.inl \
          componentmodel/behavior/MasterSolver.h \
          componentmodel/behavior/MechanicalMapping.h \
          componentmodel/behavior/MechanicalMapping.inl \
          componentmodel/behavior/MechanicalState.h \
          componentmodel/behavior/MultiVector.h \
          componentmodel/behavior/MultiMatrix.h \
          componentmodel/behavior/OdeSolver.h \
          componentmodel/behavior/State.h \
          componentmodel/collision/BroadPhaseDetection.h \
          componentmodel/collision/CollisionAlgorithm.h \
          componentmodel/collision/CollisionGroupManager.h \
          componentmodel/collision/Contact.h \
          componentmodel/collision/ContactManager.h \
          componentmodel/collision/Detection.h \
          componentmodel/collision/DetectionOutput.h \
          componentmodel/collision/Intersection.h \
          componentmodel/collision/Intersection.inl \
          componentmodel/collision/NarrowPhaseDetection.h \
          componentmodel/collision/Pipeline.h \
          componentmodel/topology/BaseMeshTopology.h \
          componentmodel/topology/BaseTopology.h \
          componentmodel/topology/BaseTopologyObject.h \
          componentmodel/topology/TopologicalMapping.h \
          componentmodel/topology/Topology.h \
          componentmodel/loader/BaseLoader.h \
          componentmodel/loader/ImageLoader.h \
          componentmodel/loader/Material.h \
          componentmodel/loader/MeshLoader.h \
          componentmodel/loader/PrimitiveGroup.h \
          Mapping.h \
          Mapping.inl \
          objectmodel/Base.h \
          objectmodel/BaseClass.h \
          objectmodel/BaseContext.h \
          objectmodel/BaseNode.h \
          objectmodel/BaseObject.h \
          objectmodel/BaseObjectDescription.h \
          objectmodel/ClassInfo.h \
          objectmodel/Context.h \
          objectmodel/ContextObject.h \
          objectmodel/Data.h \
          objectmodel/DataFileName.h \
	  objectmodel/DDGNode.h \
          objectmodel/DetachNodeEvent.h \
          objectmodel/Event.h \
          objectmodel/DataPtr.h \
          objectmodel/BaseData.h \
	  objectmodel/JoystickEvent.h \
          objectmodel/KeypressedEvent.h \
	  objectmodel/KeyreleasedEvent.h \
	  objectmodel/MouseEvent.h \
	  objectmodel/OmniEvent.h \
	  objectmodel/GLInitializedEvent.h \
          objectmodel/VDataPtr.h \
          objectmodel/Tag.h \
          objectmodel/XDataPtr.h \
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
          objectmodel/Context.cpp \
          objectmodel/Data.cpp \
	  objectmodel/DDGNode.cpp \
          objectmodel/DetachNodeEvent.cpp \
          objectmodel/Event.cpp \
          objectmodel/DataFileName.cpp \
          objectmodel/DataPtr.cpp \
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
          CollisionModel.cpp \
          DataEngine.cpp \
          componentmodel/behavior/LinearSolver.cpp \
          componentmodel/behavior/MasterSolver.cpp \
          componentmodel/behavior/MultiMatrix.cpp \
          componentmodel/behavior/OdeSolver.cpp \
          componentmodel/collision/Contact.cpp \
          componentmodel/behavior/ConstraintSolver.cpp \
          componentmodel/collision/Intersection.cpp \
          componentmodel/collision/Pipeline.cpp \
          componentmodel/topology/BaseMeshTopology.cpp \
          componentmodel/topology/BaseTopology.cpp \
          componentmodel/behavior/BaseConstraint.cpp \
          componentmodel/behavior/BaseForceField.cpp \   
          componentmodel/behavior/BaseLMConstraint.cpp \
          componentmodel/behavior/ForceField.cpp \
          componentmodel/behavior/LMConstraint.cpp \
          componentmodel/behavior/Mass.cpp \
          componentmodel/behavior/Constraint.cpp \
          componentmodel/behavior/MechanicalMapping.cpp \
          componentmodel/behavior/PairInteractionForceField.cpp \
          componentmodel/behavior/MixedInteractionForceField.cpp \
          componentmodel/behavior/PairInteractionConstraint.cpp \
          componentmodel/behavior/MixedInteractionConstraint.cpp \
          componentmodel/loader/MeshLoader.cpp 

contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV

HEADERS += \
 componentmodel/fem/BaseShapeFunction.h 
# componentmodel/fem/BaseFiniteElement.h 

SOURCES +=  \
 componentmodel/fem/BaseShapeFunction.cpp 
  
} # END SOFA_DEV


 



