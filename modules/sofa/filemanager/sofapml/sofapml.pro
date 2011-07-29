load(sofa/pre)

TEMPLATE = lib
TARGET = sofapml

DEFINES += SOFA_BUILD_FILEMANAGER_PML

HEADERS += sofapml.h \
           PMLBody.h \
           PMLRigidBody.h \
           PMLFemForceField.h \
           PMLStiffSpringForceField.h \
           PMLInteractionForceField.h \
           PMLMappedBody.h \
           PMLReader.h \
           LMLConstraint.h \
           LMLConstraint.inl \
           LMLForce.h \
           LMLForce.inl \
           LMLReader.h


SOURCES += PMLBody.cpp \
           PMLRigidBody.cpp \
           PMLFemForceField.cpp \
           PMLStiffSpringForceField.cpp \
           PMLInteractionForceField.cpp \
           PMLMappedBody.cpp \
           PMLReader.cpp \
           LMLConstraint.cpp \
           LMLForce.cpp \
           LMLReader.cpp 

load(sofa/post)
