# Target is a library:  sofadefaulttype

SOFA_DIR = ../../..
TEMPLATE = lib
include($$SOFA_DIR/sofa.cfg)

TARGET = sofadefaulttype$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES
LIBS += -lsofahelper$$LIBSUFFIX
LIBS += $$SOFA_EXT_LIBS

HEADERS += \
	  BaseMatrix.h \
	  BaseVector.h \
          Frame.h \ 
          LaparoscopicRigidTypes.h \
          Mat.h \
          NewMatSofaMatrix.h \
          NewMatSofaVector.h \
	  NewMatMatrix.h \
          NewMatVector.h \
          Quat.h \
          Quat.inl \
          RigidTypes.h \
          SofaBaseMatrix.h \
          SofaBaseVector.h \
          SolidTypes.h \
          SolidTypes.inl \
          Vec.h \
          VecTypes.h \
          Vec3Types.h

SOURCES += \
          Frame.cpp \
          Quat.cpp \
          SolidTypes.cpp

