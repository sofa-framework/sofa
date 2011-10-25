load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_forcefield

DEFINES += SOFA_BUILD_MISC_FORCEFIELD

HEADERS += initMiscForcefield.h \
           mass/MatrixMass.h \
           mass/MatrixMass.inl \
           mass/MeshMatrixMass.h \
           mass/MeshMatrixMass.inl \
           forcefield/LennardJonesForceField.h \
           forcefield/LennardJonesForceField.inl \
           forcefield/WashingMachineForceField.h \
           forcefield/WashingMachineForceField.inl \
           interactionforcefield/GearSpringForceField.h \
           interactionforcefield/GearSpringForceField.inl \
           interactionforcefield/LineBendingSprings.h \
           interactionforcefield/LineBendingSprings.inl

SOURCES += initMiscForcefield.cpp \
           mass/MatrixMass.cpp \
           mass/MeshMatrixMass.cpp \
           forcefield/LennardJonesForceField.cpp \
           forcefield/WashingMachineForceField.cpp \
           interactionforcefield/GearSpringForceField.cpp \
           interactionforcefield/LineBendingSprings.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
