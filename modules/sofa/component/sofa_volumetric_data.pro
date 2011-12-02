load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_volumetric_data

DEFINES += SOFA_BUILD_VOLUMETRIC_DATA

HEADERS += initVolumetricData.h \
           container/ImplicitSurfaceContainer.h \
           container/InterpolatedImplicitSurface.h \
           container/InterpolatedImplicitSurface.inl \
           forcefield/DistanceGridForceField.h \
           forcefield/DistanceGridForceField.inl \
           mapping/ImplicitSurfaceMapping.h \
           mapping/ImplicitSurfaceMapping.inl \
           container/DistanceGrid.h \
           collision/DistanceGridCollisionModel.h \
           collision/RigidDistanceGridDiscreteIntersection.h \
           collision/RigidDistanceGridDiscreteIntersection.inl \
           collision/FFDDistanceGridDiscreteIntersection.h \
           collision/FFDDistanceGridDiscreteIntersection.inl \

SOURCES += initVolumetricData.cpp \
           container/ImplicitSurfaceContainer.cpp \
           container/InterpolatedImplicitSurface.cpp \
           forcefield/DistanceGridForceField.cpp \
           mapping/ImplicitSurfaceMapping.cpp \
           container/DistanceGrid.cpp \
           collision/DistanceGridCollisionModel.cpp \
           collision/RayDistanceGridContact.cpp \
           collision/RigidDistanceGridDiscreteIntersection.cpp \
           collision/FFDDistanceGridDiscreteIntersection.cpp \
           collision/BarycentricPenalityContact_DistanceGrid.cpp \

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
