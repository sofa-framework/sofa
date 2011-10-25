load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_validation

DEFINES += SOFA_BUILD_VALIDATION

HEADERS += initValidation.h \
           misc/CompareState.h \
           misc/CompareTopology.h \
           misc/DevAngleCollisionMonitor.h \
           misc/DevAngleCollisionMonitor.inl \
           misc/DevTensionMonitor.h \
           misc/DevTensionMonitor.inl \
           misc/DevMonitorManager.h \
           misc/ExtraMonitor.h \
           misc/ExtraMonitor.inl \
           misc/Monitor.h \
           misc/Monitor.inl \
           misc/EvalPointsDistance.h \
           misc/EvalPointsDistance.inl \
           misc/EvalSurfaceDistance.h \
           misc/EvalSurfaceDistance.inl

SOURCES += initValidation.cpp \
           misc/CompareState.cpp \
           misc/CompareTopology.cpp \
           misc/DevAngleCollisionMonitor.cpp \
           misc/DevTensionMonitor.cpp \
           misc/DevMonitorManager.cpp \
           misc/ExtraMonitor.cpp \
           misc/Monitor.cpp \
           misc/EvalPointsDistance.cpp \
           misc/EvalSurfaceDistance.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
