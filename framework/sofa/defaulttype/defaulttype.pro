# Target is a library:  sofadefaulttype

SOFA_DIR = ../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = sofadefaulttype$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES
LIBS += -lsofahelper$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_DIR/modules
INCLUDEPATH -= $$SOFA_DIR/applications

HEADERS += \
	  BaseMatrix.h \
	  BaseVector.h \
	  DataTypeInfo.h \
          Frame.h \ 
          LaparoscopicRigidTypes.h \
          Mat.h \
	  NewMatMatrix.h \
          NewMatVector.h \
          Quat.h \
          Quat.inl \
          RigidTypes.h \
          SolidTypes.h \
          SolidTypes.inl \
          Vec.h \
          VecTypes.h \
          Vec3Types.h

SOURCES += \
          Frame.cpp \
          SolidTypes.cpp

contains(DEFINES,SOFA_HAVE_MKL) {

HEADERS += \
	MKLMatrix.h \
	MKLVector.h

}
