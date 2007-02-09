# Target is a library:  sofahelper

SOFA_DIR = ../../..
TEMPLATE = lib
include($$SOFA_DIR/sofa.cfg)

TARGET = sofahelper$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES qt uic3
QT += opengl qt3support
LIBS = -lsofadefaulttype$$LIBSUFFIX 
win32{
  LIBS += -llibxml2 -lGLaux -lglut32 -lopengl32 -lglu32
  LIBS += -llibxml2 -lGLaux -lglut32 -lopengl32 -lglu32
}
unix{
  QMAKE_LIBDIR += /usr/X11R6/lib
  LIBS += -lglut -lGL -lGLU -lpthread -lxml2 -lz
  QMAKE_LIBDIR += /usr/X11R6/lib
  LIBS += -lglut -lGL -lGLU -lpthread -lxml2 -lz
}
contains(DEFINES,SOFA_HAVE_PNG){
  LIBS += -lpng -lz
  LIBS += -lpng -lz
}

HEADERS += \
          ArgumentParser.h \
          BackTrace.h \
          fixed_array.h \
          Factory.h \
          Factory.inl \
          FnDispatcher.h \
          FnDispatcher.inl \
          gl/Axis.h \
          gl/Capture.h \
          gl/glfont.h \
          gl/GLshader.h \
          gl/RAII.h \
          gl/template.h \
          gl/Texture.h \
          gl/Trackball.h \
          gl/Transformation.h \
          io/Image.h \
          io/ImageBMP.h \
          io/ImagePNG.h \
          io/MassSpringLoader.h \
          io/Mesh.h \
          io/MeshOBJ.h \
          io/MeshTopologyLoader.h \
          io/MeshTrian.h \
          io/SphereLoader.h \
          io/TriangleLoader.h \
          LCPSolver.h \
          LCPSolver.inl \
          PolarDecompose.h \
          rmath.h \
          static_assert.h \
          system/config.h \
          system/SetDirectory.h \
          system/thread/CTime.h \
          system/thread/debug.h \
          vector.h 
          
SOURCES += \
          ArgumentParser.cpp \
          BackTrace.cpp \
          FnDispatcher.cpp \
          Factory.cpp \
          gl/Axis.cpp \
          gl/Capture.cpp \
          gl/glfont.cpp \
          gl/GLshader.cpp \
          gl/Texture.cpp \
          gl/Trackball.cpp \
          gl/Transformation.cpp \
          io/Image.cpp \
          io/ImageBMP.cpp \
          io/ImagePNG.cpp \
          io/MassSpringLoader.cpp \
          io/Mesh.cpp \
          io/MeshOBJ.cpp \
          io/MeshTopologyLoader.cpp \
          io/MeshTrian.cpp \
          io/SphereLoader.cpp \
          io/TriangleLoader.cpp \
          system/SetDirectory.cpp \
          system/thread/CTime.cpp \
          system/thread/debug.cpp
