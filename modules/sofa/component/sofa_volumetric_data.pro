load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_volumetric_data

DEFINES += SOFA_BUILD_VOLUMETRIC_DATA

HEADERS += container/ImplicitSurfaceContainer.h \
           container/InterpolatedImplicitSurface.h \
           container/InterpolatedImplicitSurface.inl \
           forcefield/DistanceGridForceField.h \
           forcefield/DistanceGridForceField.inl \
           mapping/ImplicitSurfaceMapping.h \
           mapping/ImplicitSurfaceMapping.inl

SOURCES += container/ImplicitSurfaceContainer.cpp \
           container/InterpolatedImplicitSurface.cpp \
           forcefield/DistanceGridForceField.cpp \
           mapping/ImplicitSurfaceMapping.cpp


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
