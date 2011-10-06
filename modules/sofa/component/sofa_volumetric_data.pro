load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_volumetric_data

DEFINES += SOFA_BUILD_VOLUMETRIC_DATA

HEADERS += collision/DistanceGridCollisionModel.h \
           container/ImplicitSurfaceContainer.h \
           container/InterpolatedImplicitSurface.h \
           container/InterpolatedImplicitSurface.inl \
           container/DistanceGrid.h \
           forcefield/DistanceGridForceField.h \
           forcefield/DistanceGridForceField.inl \
           mapping/ImplicitSurfaceMapping.h \
           mapping/ImplicitSurfaceMapping.inl

SOURCES += collision/DistanceGridCollisionModel.cpp \
           container/ImplicitSurfaceContainer.cpp \
           container/InterpolatedImplicitSurface.cpp \
           container/DistanceGrid.cpp \
           forcefield/DistanceGridForceField.cpp \
           mapping/ImplicitSurfaceMapping.cpp


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
