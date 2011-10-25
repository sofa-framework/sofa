# Target is a library:  sofadefaulttype
load(sofa/pre)

TEMPLATE = lib
TARGET = sofadefaulttype

DEFINES += SOFA_BUILD_DEFAULTTYPE
# Make sure there are no cross-dependencies
INCLUDEPATH -= $$ROOT_SRC_DIR/framework/sofa/core
DEPENDPATH -= $$ROOT_SRC_DIR/framework/sofa/core
INCLUDEPATH -= $$ROOT_SRC_DIR/modules
DEPENDPATH -= $$ROOT_SRC_DIR/modules
INCLUDEPATH -= $$ROOT_SRC_DIR/applications
DEPENDPATH -= $$ROOT_SRC_DIR/applications

HEADERS += \
        defaulttype.h \
	  BaseMatrix.h \
	  BaseVector.h \
	  BoundingBox.h \
	  DataTypeInfo.h \
          Frame.h \ 
          LaparoscopicRigidTypes.h \
          MapMapSparseMatrix.h \
          Mat.h \
          Quat.h \
          Quat.inl \
          #RigidInertia.h \
          #RigidInertia.inl \
          RigidTypes.h \
          RigidVec6Types.h \
          SolidTypes.h \
          SolidTypes.inl \
          #SparseConstraintTypes.h \
          Vec.h \
          VecTypes.h \
	    MatSym.h \
          Vec3Types.h

SOURCES += \
          BaseMatrix.cpp \
		  BoundingBox.cpp \
          Frame.cpp \
          #RigidInertia.cpp \
          SolidTypes.cpp

load(sofa/post)
