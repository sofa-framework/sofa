load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_sph_fluid

DEFINES += SOFA_BUILD_SPH_FLUID

HEADERS += initSPHFluid.h \
           container/SpatialGridContainer.h \
           container/SpatialGridContainer.inl \
           forcefield/SPHFluidForceField.h \
           forcefield/SPHFluidForceField.inl \
           mapping/SPHFluidSurfaceMapping.h \
           mapping/SPHFluidSurfaceMapping.inl \
           misc/ParticleSink.h \
           misc/ParticleSource.h \
           forcefield/ParticlesRepulsionForceField.h \
           forcefield/ParticlesRepulsionForceField.inl

SOURCES += initSPHFluid.cpp \
           container/SpatialGridContainer.cpp \
           forcefield/SPHFluidForceField.cpp \
           mapping/SPHFluidSurfaceMapping.cpp \
           misc/ParticleSink.cpp \
           misc/ParticleSource.cpp \
           forcefield/ParticlesRepulsionForceField.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
