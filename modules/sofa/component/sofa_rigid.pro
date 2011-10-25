load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_rigid

DEFINES += SOFA_BUILD_RIGID

HEADERS += initRigid.h \
           container/ArticulatedHierarchyContainer.h \
           container/ArticulatedHierarchyContainer.inl \
           mapping/ArticulatedSystemMapping.h \
           mapping/ArticulatedSystemMapping.inl \
           mapping/LaparoscopicRigidMapping.h \
           mapping/LaparoscopicRigidMapping.inl \
           mapping/LineSetSkinningMapping.h \
           mapping/LineSetSkinningMapping.inl \
           mapping/RigidMapping.h \
           mapping/RigidMapping.inl \
           mapping/RigidRigidMapping.h \
           mapping/RigidRigidMapping.inl \
           mapping/SkinningMapping.h \
           mapping/SkinningMapping.inl \
           interactionforcefield/JointSpringForceField.h \
           interactionforcefield/JointSpringForceField.inl

SOURCES += initRigid.cpp \
           container/ArticulatedHierarchyContainer.cpp \
           mapping/ArticulatedSystemMapping.cpp \
           mapping/LaparoscopicRigidMapping.cpp \
           mapping/LineSetSkinningMapping.cpp \
           mapping/RigidMapping.cpp \
           mapping/RigidRigidMapping.cpp \
           mapping/SkinningMapping.cpp \
           interactionforcefield/JointSpringForceField.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
