load(sofa/pre)

TEMPLATE = lib
TARGET = sofa_loader

DEFINES += SOFA_BUILD_LOADER

HEADERS += initLoader.h \
           loader/MeshGmshLoader.h \
           loader/MeshObjLoader.h \
           loader/MeshOffLoader.h \
           loader/MeshTrianLoader.h \
           loader/MeshVTKLoader.h \
           loader/MeshSTLLoader.h \
           loader/MeshXspLoader.h \
           loader/OffSequenceLoader.h \
           loader/SphereLoader.h \
           loader/VoxelGridLoader.h \
           misc/InputEventReader.h \
           misc/ReadState.h \
           misc/ReadState.inl \
           misc/ReadTopology.h \
           misc/ReadTopology.inl 

SOURCES += initLoader.cpp \
           loader/MeshGmshLoader.cpp \
           loader/MeshObjLoader.cpp \
           loader/MeshOffLoader.cpp \
           loader/MeshTrianLoader.cpp \
           loader/MeshVTKLoader.cpp \
           loader/MeshSTLLoader.cpp \
           loader/MeshXspLoader.cpp \
           loader/OffSequenceLoader.cpp \
           loader/SphereLoader.cpp \
           loader/VoxelGridLoader.cpp \
           misc/InputEventReader.cpp \
           misc/ReadState.cpp \
           misc/ReadTopology.cpp

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_INSTALL_INC_DIR/applications
DEPENDPATH -= $$SOFA_INSTALL_INC_DIR/applications

#exists(component-local.cfg): include(component-local.cfg)

load(sofa/post)
 
