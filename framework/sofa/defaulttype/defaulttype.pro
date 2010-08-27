# Target is a library:  sofadefaulttype

SOFA_DIR = ../../..
TEMPLATE = lib
TARGET = sofadefaulttype

include($${SOFA_DIR}/sofa.cfg)

CONFIG += $$CONFIGLIBRARIES

!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}

LIBS += -lsofahelper$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS
DEFINES += SOFA_BUILD_DEFAULTTYPE
# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_DIR/modules
INCLUDEPATH -= $$SOFA_DIR/applications

HEADERS += \
        defaulttype.h \
	  BaseMatrix.h \
	  BaseVector.h \
	  DataTypeInfo.h \
          Frame.h \ 
          LaparoscopicRigidTypes.h \
		  MapMapSparseMatrix.h \
          Mat.h \
          Quat.h \
          Quat.inl \
          RigidTypes.h \
          SolidTypes.h \
          SolidTypes.inl \
          SparseConstraintTypes.h \
          Vec.h \
          VecTypes.h \
	    MatSym.h \
          Vec3Types.h

SOURCES += \
          BaseMatrix.cpp \
          Frame.cpp \
          SolidTypes.cpp

contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV

HEADERS += AffineTypes.h \
           FrameMass.h \
           QuadraticTypes.h

SOURCES += 

} # END SOFA_DEV

