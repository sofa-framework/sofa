load(sofa/pre)
defineAsApp(SofaPhysicsAPI)

TEMPLATE = app
TARGET = SofaPhysicsAPI

SOURCES += main.cpp \
    SofaPhysicsSimulation.cpp \
    fakegui.cpp

HEADERS += \
    SofaPhysicsSimulation.h \
    SofaPhysicsSimulation_impl.h \
    fakegui.h

#macx {
#	CONFIG += app_bundle
#	ICON = sofa-hms1.icns
#} else {
#	RC_FILE = sofa.rc
#}

load(sofa/post)
