load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_mapping

# with gcc 4.5.2, PCAOnRigidFrameMapping generates an Internal Compiler Error when using the highest optimization level
QMAKE_CFLAGS_RELEASE   -= -O3
QMAKE_CXXFLAGS_RELEASE -= -O3
QMAKE_CFLAGS_RELEASE   += -O2
QMAKE_CXXFLAGS_RELEASE += -O2

DEFINES *= EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET


DEFINES += SOFA_BUILD_MISC_MAPPING

win32 {
	QMAKE_CXXFLAGS += /bigobj
}

HEADERS += initMiscMapping.h \
           mapping/BeamLinearMapping.h \
           mapping/BeamLinearMapping.inl \
           mapping/CenterPointMechanicalMapping.h \
           mapping/CenterPointMechanicalMapping.inl \
           mapping/CenterOfMassMapping.h \
           mapping/CenterOfMassMapping.inl \
           mapping/CenterOfMassMultiMapping.h \
           mapping/CenterOfMassMultiMapping.inl \
           mapping/CenterOfMassMulti2Mapping.h \
           mapping/CenterOfMassMulti2Mapping.inl \
           mapping/CurveMapping.h \
           mapping/CurveMapping.inl \
           mapping/ExternalInterpolationMapping.h \
           mapping/ExternalInterpolationMapping.inl \
           mapping/SubsetMultiMapping.h \
           mapping/SubsetMultiMapping.inl \
           mapping/TubularMapping.h \
           mapping/TubularMapping.inl \
           mapping/VoidMapping.h \
           mapping/BarycentricMappingRigid.h \
           mapping/BarycentricMappingRigid.inl \
           mapping/DeformableOnRigidFrameMapping.h \
           mapping/DeformableOnRigidFrameMapping.inl \
           mapping/PCAOnRigidFrameMapping.h \
           mapping/PCAOnRigidFrameMapping.inl \
           mapping/CatmullRomSplineMapping.h \
           mapping/CatmullRomSplineMapping.inl

SOURCES += initMiscMapping.cpp \
           mapping/BeamLinearMapping.cpp \
           mapping/CenterPointMechanicalMapping.cpp \
           mapping/CenterOfMassMapping.cpp \
           mapping/CenterOfMassMultiMapping.cpp \
           mapping/CenterOfMassMulti2Mapping.cpp \
           mapping/CurveMapping.cpp \
           mapping/ExternalInterpolationMapping.cpp \
           mapping/SubsetMultiMapping.cpp \
           mapping/TubularMapping.cpp \
           mapping/VoidMapping.cpp \
           mapping/BarycentricMappingRigid.cpp \
           mapping/DeformableOnRigidFrameMapping.cpp \
           mapping/PCAOnRigidFrameMapping.cpp \
           mapping/CatmullRomSplineMapping.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
