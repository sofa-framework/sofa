load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_boundary_condition

DEFINES += SOFA_BUILD_BOUNDARY_CONDITION

HEADERS += initBoundaryCondition.h \
           forcefield/AspirationForceField.h \
           forcefield/AspirationForceField.inl \
           forcefield/BuoyantForceField.h \
           forcefield/BuoyantForceField.inl \
           forcefield/ConicalForceField.h \
           forcefield/ConicalForceField.inl \
           forcefield/ConstantForceField.h \
           forcefield/ConstantForceField.inl \
           forcefield/EdgePressureForceField.h \
           forcefield/EdgePressureForceField.inl \
           forcefield/EllipsoidForceField.h \
           forcefield/EllipsoidForceField.inl \
           forcefield/LinearForceField.h \
           forcefield/LinearForceField.inl \
           forcefield/OscillatingTorsionPressureForceField.h \
           forcefield/OscillatingTorsionPressureForceField.inl \
           forcefield/PlaneForceField.h \
           forcefield/PlaneForceField.inl \
           forcefield/SphereForceField.h \
           forcefield/SphereForceField.inl \
           forcefield/SurfacePressureForceField.h \
           forcefield/SurfacePressureForceField.inl \
           forcefield/TrianglePressureForceField.h \
           forcefield/TrianglePressureForceField.inl \
           forcefield/VaccumSphereForceField.h \
           forcefield/VaccumSphereForceField.inl \
           projectiveconstraintset/FixedConstraint.h \
           projectiveconstraintset/FixedConstraint.inl \
           projectiveconstraintset/FixedPlaneConstraint.h \
           projectiveconstraintset/FixedPlaneConstraint.inl \
           projectiveconstraintset/FixedRotationConstraint.h \
           projectiveconstraintset/FixedRotationConstraint.inl \
           projectiveconstraintset/FixedTranslationConstraint.h \
           projectiveconstraintset/FixedTranslationConstraint.inl \
           projectiveconstraintset/HermiteSplineConstraint.h \
           projectiveconstraintset/HermiteSplineConstraint.inl \
           projectiveconstraintset/LinearMovementConstraint.h \
           projectiveconstraintset/LinearMovementConstraint.inl \
           projectiveconstraintset/LinearVelocityConstraint.h \
           projectiveconstraintset/LinearVelocityConstraint.inl \
           projectiveconstraintset/OscillatorConstraint.h \
           projectiveconstraintset/OscillatorConstraint.inl \
           projectiveconstraintset/ParabolicConstraint.h \
           projectiveconstraintset/ParabolicConstraint.inl \
           projectiveconstraintset/PartialFixedConstraint.h \
           projectiveconstraintset/PartialFixedConstraint.inl \
           projectiveconstraintset/PartialLinearMovementConstraint.h \
           projectiveconstraintset/PartialLinearMovementConstraint.inl \
           projectiveconstraintset/PositionBasedDynamicsConstraint.h \
           projectiveconstraintset/PositionBasedDynamicsConstraint.inl


SOURCES += initBoundaryCondition.cpp \
           forcefield/AspirationForceField.cpp \
           forcefield/BuoyantForceField.cpp \
           forcefield/ConicalForceField.cpp \
           forcefield/ConstantForceField.cpp \
           forcefield/EdgePressureForceField.cpp \
           forcefield/EllipsoidForceField.cpp \
           forcefield/LinearForceField.cpp \
           forcefield/OscillatingTorsionPressureForceField.cpp \
           forcefield/PlaneForceField.cpp \
           forcefield/SphereForceField.cpp \
           forcefield/SurfacePressureForceField.cpp \
           forcefield/TrianglePressureForceField.cpp \
           forcefield/VaccumSphereForceField.cpp \
           projectiveconstraintset/FixedConstraint.cpp \
           projectiveconstraintset/FixedPlaneConstraint.cpp \
           projectiveconstraintset/FixedRotationConstraint.cpp \
           projectiveconstraintset/FixedTranslationConstraint.cpp \
           projectiveconstraintset/HermiteSplineConstraint.cpp \
           projectiveconstraintset/LinearMovementConstraint.cpp \
           projectiveconstraintset/LinearVelocityConstraint.cpp \
           projectiveconstraintset/OscillatorConstraint.cpp \
           projectiveconstraintset/ParabolicConstraint.cpp \
           projectiveconstraintset/PartialFixedConstraint.cpp \
           projectiveconstraintset/PartialLinearMovementConstraint.cpp \
           projectiveconstraintset/PositionBasedDynamicsConstraint.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
