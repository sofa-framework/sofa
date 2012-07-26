load(sofa/pre)
defineAsApp(SofaPhysicsAPI)

TEMPLATE = app
TARGET = SofaPhysicsAPI

SOURCES += main.cpp \
    SofaPhysicsSimulation.cpp \
    SofaPhysicsOutputMesh.cpp \
    SofaPhysicsDataMonitor.cpp \
    SofaPhysicsDataController.cpp \
    fakegui.cpp

HEADERS += \
    SofaPhysicsAPI.h \
    SofaPhysicsSimulation_impl.h \
    SofaPhysicsOutputMesh_impl.h \
    SofaPhysicsDataMonitor_impl.h \
    SofaPhysicsDataController_impl.h \
    fakegui.h

#macx {
#	CONFIG += app_bundle
#	ICON = sofa-hms1.icns
#} else {
#	RC_FILE = sofa.rc
#}

load(sofa/post)
