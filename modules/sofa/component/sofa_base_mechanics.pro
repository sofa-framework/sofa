load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_base_mechanics

DEFINES += SOFA_BUILD_BASE_MECHANICS

HEADERS += initBaseMechanics.h \
           container/MappedObject.h \
           container/MappedObject.inl \
           container/MechanicalObject.h \
           container/MechanicalObject.inl \
           mass/AddMToMatrixFunctor.h \
           mass/DiagonalMass.h \
           mass/DiagonalMass.inl \
           mass/UniformMass.h \
           mass/UniformMass.inl \
           mapping/BarycentricMapping.h \
           mapping/BarycentricMapping.inl \
           mapping/IdentityMapping.h \
           mapping/IdentityMapping.inl \
           mapping/SubsetMapping.h \
           mapping/SubsetMapping.inl

SOURCES += initBaseMechanics.cpp \
           container/MappedObject.cpp \
           container/MechanicalObject.cpp \
           mass/DiagonalMass.cpp \
           mass/UniformMass.cpp \
           mapping/BarycentricMapping.cpp \
           mapping/IdentityMapping.cpp \
           mapping/SubsetMapping.cpp

contains(DEFINES,SOFA_SMP){
HEADERS += container/MechanicalObjectTasks.inl
}

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
