load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_haptics

DEFINES += SOFA_BUILD_HAPTICS

HEADERS += controller/ForceFeedback.h \
           controller/NullForceFeedbackT.h \
           controller/NullForceFeedback.h \
           controller/EnslavementForceFeedback.h \
           controller/LCPForceFeedback.h \
           controller/LCPForceFeedback.inl \
           controller/MechanicalStateForceFeedback.h

SOURCES += controller/NullForceFeedback.cpp \
           controller/NullForceFeedbackT.cpp \
           controller/EnslavementForceFeedback.cpp \
           controller/LCPForceFeedback.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
