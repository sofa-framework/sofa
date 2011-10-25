load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_object_interaction

DEFINES += SOFA_BUILD_OBJECT_INTERACTION

HEADERS += initObjectInteraction.h \
           projectiveconstraintset/AttachConstraint.h \
           projectiveconstraintset/AttachConstraint.inl \
           interactionforcefield/BoxStiffSpringForceField.h \
           interactionforcefield/BoxStiffSpringForceField.inl \
           interactionforcefield/InteractionEllipsoidForceField.h \
           interactionforcefield/InteractionEllipsoidForceField.inl \
           interactionforcefield/PenalityContactForceField.h \
           interactionforcefield/PenalityContactForceField.inl \
           interactionforcefield/RepulsiveSpringForceField.h \
           interactionforcefield/RepulsiveSpringForceField.inl


SOURCES += initObjectInteraction.cpp \
           projectiveconstraintset/AttachConstraint.cpp \
           interactionforcefield/BoxStiffSpringForceField.cpp \
           interactionforcefield/InteractionEllipsoidForceField.cpp \
           interactionforcefield/PenalityContactForceField.cpp \
           interactionforcefield/RepulsiveSpringForceField.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
