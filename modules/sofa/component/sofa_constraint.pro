load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_constraint

DEFINES += SOFA_BUILD_CONSTRAINT

HEADERS += collision/LocalMinDistance.h \
           collision/LocalMinDistance.inl \
           collision/LMDNewProximityIntersection.h \
           collision/LMDNewProximityIntersection.inl \
           collision/LineLocalMinDistanceFilter.h \
           collision/LocalMinDistanceFilter.h \
           collision/PointLocalMinDistanceFilter.h \
           collision/TriangleLocalMinDistanceFilter.h \
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

SOURCES += collision/LocalMinDistance.cpp \
           collision/LMDNewProximityIntersection.cpp \
           collision/LineLocalMinDistanceFilter.cpp \
           collision/PointLocalMinDistanceFilter.cpp \
           collision/TriangleLocalMinDistanceFilter.cpp \
           collision/FrictionContact.cpp \
           constraintset/UnilateralInteractionConstraint.cpp \
           constraintset/UncoupledConstraintCorrection.cpp \
           constraintset/PrecomputedConstraintCorrection.cpp \
           constraintset/LinearSolverConstraintCorrection.cpp \
           constraintset/LCPConstraintSolver.cpp \
           constraintset/ConstraintSolverImpl.cpp \
           animationloop/FreeMotionAnimationLoop.cpp \

contains(DEFINES,SOFA_DEV){
HEADERS += collision/BSplineModel.h \
           collision/BSplineModel.inl
SOURCES += collision/BSplineModel.cpp
}

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

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
