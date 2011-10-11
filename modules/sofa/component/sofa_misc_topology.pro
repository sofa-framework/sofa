load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_misc_topology

DEFINES += SOFA_BUILD_MISC_TOPOLOGY
DEFINES += POINT_DATA_VECTOR_ACCESS

HEADERS += misc/TopologicalChangeProcessor.h

SOURCES += misc/TopologicalChangeProcessor.cpp


# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
