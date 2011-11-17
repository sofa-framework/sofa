load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_constraint

DEFINES += SOFA_BUILD_CONSTRAINT

HEADERS += initConstraint.h \
           collision/LocalMinDistance.h \
#           collision/LocalMinDistance.inl \
           collision/LMDNewProximityIntersection.h \
           collision/LMDNewProximityIntersection.inl \
           collision/FrictionContact.h \
           collision/FrictionContact.inl \
           constraintset/UnilateralInteractionConstraint.h \
           constraintset/UnilateralInteractionConstraint.inl \
           constraintset/UncoupledConstraintCorrection.h \
           constraintset/UncoupledConstraintCorrection.inl \
           constraintset/PrecomputedConstraintCorrection.h \
           constraintset/PrecomputedConstraintCorrection.inl \
           constraintset/LinearSolverConstraintCorrection.h \
           constraintset/LinearSolverConstraintCorrection.inl \
           constraintset/LCPConstraintSolver.h \
           constraintset/ConstraintSolverImpl.h \
           animationloop/FreeMotionAnimationLoop.h \
           constraintset/ContactDescription.h \
           constraintset/GenericConstraintSolver.h \
           constraintset/BilateralInteractionConstraint.h \
           constraintset/BilateralInteractionConstraint.inl \
           animationloop/ConstraintAnimationLoop.h \
           constraintset/StopperConstraint.h \
           constraintset/StopperConstraint.inl \
           constraintset/SlidingConstraint.h \
           constraintset/SlidingConstraint.inl \


SOURCES += initConstraint.cpp \
           collision/LocalMinDistance.cpp \
           collision/LMDNewProximityIntersection.cpp \
           collision/FrictionContact.cpp \
           constraintset/UnilateralInteractionConstraint.cpp \
           constraintset/UncoupledConstraintCorrection.cpp \
           constraintset/PrecomputedConstraintCorrection.cpp \
           constraintset/LinearSolverConstraintCorrection.cpp \
           constraintset/LCPConstraintSolver.cpp \
           constraintset/ConstraintSolverImpl.cpp \
           animationloop/FreeMotionAnimationLoop.cpp \
           constraintset/GenericConstraintSolver.cpp \
           constraintset/BilateralInteractionConstraint.cpp \
           animationloop/ConstraintAnimationLoop.cpp \
           constraintset/StopperConstraint.cpp \
           constraintset/SlidingConstraint.cpp \


contains(DEFINES,SOFA_HAVE_EIGEN2){
HEADERS += collision/BarycentricDistanceLMConstraintContact.h \
           collision/BarycentricDistanceLMConstraintContact.inl \
           constraintset/DOFBlockerLMConstraint.h \
           constraintset/DOFBlockerLMConstraint.inl \
           constraintset/FixedLMConstraint.h \
           constraintset/FixedLMConstraint.inl \
           constraintset/DistanceLMContactConstraint.h \
           constraintset/DistanceLMContactConstraint.inl \
           constraintset/DistanceLMConstraint.h \
           constraintset/DistanceLMConstraint.inl \
           constraintset/LMConstraintSolver.h \
           constraintset/LMConstraintDirectSolver.h


SOURCES += collision/BarycentricDistanceLMConstraintContact.cpp \
           constraintset/DOFBlockerLMConstraint.cpp \
           constraintset/FixedLMConstraint.cpp \
           constraintset/DistanceLMContactConstraint.cpp \
           constraintset/DistanceLMConstraint.cpp \
           constraintset/LMConstraintSolver.cpp \
           constraintset/LMConstraintDirectSolver.cpp
}

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
