load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_deformable

DEFINES += SOFA_BUILD_DEFORMABLE

HEADERS += initDeformable.h \
           forcefield/QuadularBendingSprings.h \
           forcefield/QuadularBendingSprings.inl \
           forcefield/RestShapeSpringsForceField.h \
           forcefield/RestShapeSpringsForceField.inl \
           forcefield/TriangularBendingSprings.h \
           forcefield/TriangularBendingSprings.inl \
           forcefield/TriangularBiquadraticSpringsForceField.h \
           forcefield/TriangularBiquadraticSpringsForceField.inl \
           forcefield/TriangularQuadraticSpringsForceField.h \
           forcefield/TriangularQuadraticSpringsForceField.inl \
           forcefield/TriangularTensorMassForceField.h \
           forcefield/TriangularTensorMassForceField.inl \
           interactionforcefield/FrameSpringForceField.h \
           interactionforcefield/FrameSpringForceField.inl \
           interactionforcefield/MeshSpringForceField.h \
           interactionforcefield/MeshSpringForceField.inl \
           interactionforcefield/QuadBendingSprings.h \
           interactionforcefield/QuadBendingSprings.inl \
           interactionforcefield/RegularGridSpringForceField.h \
           interactionforcefield/RegularGridSpringForceField.inl \
           interactionforcefield/SpringForceField.h \
           interactionforcefield/SpringForceField.inl \
           interactionforcefield/StiffSpringForceField.h \
           interactionforcefield/StiffSpringForceField.inl \
           interactionforcefield/TriangleBendingSprings.h \
           interactionforcefield/TriangleBendingSprings.inl \
           interactionforcefield/VectorSpringForceField.h \
           interactionforcefield/VectorSpringForceField.inl

SOURCES += initDeformable.cpp \
           forcefield/QuadularBendingSprings.cpp \
           forcefield/RestShapeSpringsForceField.cpp \
           forcefield/TriangularBendingSprings.cpp \
           forcefield/TriangularBiquadraticSpringsForceField.cpp \
           forcefield/TriangularQuadraticSpringsForceField.cpp \
           forcefield/TriangularTensorMassForceField.cpp \
           interactionforcefield/FrameSpringForceField.cpp \
           interactionforcefield/MeshSpringForceField.cpp \
           interactionforcefield/QuadBendingSprings.cpp \
           interactionforcefield/RegularGridSpringForceField.cpp \
           interactionforcefield/SpringForceField.cpp \
           interactionforcefield/StiffSpringForceField.cpp \
           interactionforcefield/TriangleBendingSprings.cpp \
           interactionforcefield/VectorSpringForceField.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
