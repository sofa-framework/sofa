# Target is a library:  sofadefaulttype

SOFA_DIR = ../../..
TEMPLATE = lib
include($$SOFA_DIR/sofa.cfg)

TARGET = sofadefaulttype$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES

HEADERS += \
          Frame.h \ 
          LaparoscopicRigidTypes.h \
          Mat.h \
          NewMatSofaMatrix.h \
          NewMatSofaVector.h \
          Quat.h \
          Quat.inl \
          RigidTypes.h \
          SofaBaseMatrix.h \
          SofaBaseVector.h \
          SolidTypes.h \
          SolidTypes.inl \
          Vec.h \ 
          Vec3Types.h 
          
SOURCES += \
          Frame.cpp \
          Quat.cpp \
          SolidTypes.cpp 
