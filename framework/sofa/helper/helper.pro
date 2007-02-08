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
          ./ArgumentParser.h \
          ./BackTrace.h \
          ./fixed_array.h \
          ./Factory.h \
          ./Factory.inl \
          ./FnDispatcher.h \
          ./FnDispatcher.inl \
          ./gl/Axis.h \
          ./gl/Capture.h \
          ./gl/glfont.h \
          ./gl/GLshader.h \
          ./gl/RAII.h \
          ./gl/template.h \
          ./gl/Texture.h \
          ./gl/Trackball.h \
          ./gl/Transformation.h \
          ./io/Encoding.h \
          ./io/Encoding.inl \
          ./io/Image.h \
          ./io/ImageBMP.h \
          ./io/ImagePNG.h \
          ./io/MassSpringLoader.h \
          ./io/Mesh.h \
          ./io/MeshOBJ.h \
          ./io/MeshTopologyLoader.h \
          ./io/MeshTrian.h \
          ./io/SphereLoader.h \
          ./io/TriangleLoader.h \
          ./LCPSolver.h \
          ./LCPSolver.inl \
          ./MultiVector.h \
          ./PolarDecompose.h \
          ./proximity.h \
          ./rmath.h \
          ./slc/bfast.h \
          ./slc/bfastUtil.h \
          ./slc/bfastVector.h \
          ./slc/distTree.h \
          ./slc/pcube/pcube.h \
          ./slc/pcube/vec.h \
          ./slc/slcSurface.h \
          ./static_assert.h \
          ./system/config.h \
          ./system/SetDirectory.h \
          ./system/thread/AutomateUtils.h \
          ./system/thread/CTime.h \
          ./system/thread/debug.h \
          ./system/thread/Edge.h \
          ./system/thread/EdgeGFX.h \
          ./system/thread/ExecBus.h \
          ./system/thread/NodeGFX.h \
          ./system/thread/ObjSubAutomate.h \
          ./system/thread/ObjSubAutomateAttributeCondExec.h \
          ./system/thread/ObjSubAutomateAttributeCondExec.inl \
          ./system/thread/ObjSubAutomateAttributeNodeExec.h \
          ./system/thread/ObjSubAutomateAttributeNodeExec.inl \
          ./system/thread/ObjSubAutomateCondExec.h \
          ./system/thread/ObjSubAutomateNodeExec.h \
          ./system/thread/StateMachine.h \
          ./system/thread/ThreadSimulation.h \
          ./system/thread/utils.h \
          ./system/thread/video.h \
          ./vector.h 
          
SOURCES += \
          ./ArgumentParser.cpp \
          ./BackTrace.cpp \
          ./FnDispatcher.cpp \
          ./Factory.cpp \
          ./gl/Axis.cpp \
          ./gl/Capture.cpp \
          ./gl/glfont.cpp \
          ./gl/GLshader.cpp \
          ./gl/Texture.cpp \
          ./gl/Trackball.cpp \
          ./gl/Transformation.cpp \
          ./io/Image.cpp \
          ./io/ImageBMP.cpp \
          ./io/ImagePNG.cpp \
          ./io/MassSpringLoader.cpp \
          ./io/Mesh.cpp \
          ./io/MeshOBJ.cpp \
          ./io/MeshTopologyLoader.cpp \
          ./io/MeshTrian.cpp \
          ./io/SphereLoader.cpp \
          ./io/TriangleLoader.cpp \
          ./proximity.cpp \
          ./slc/bfastUtil.cpp \
          ./slc/bfastVector.cpp \
          ./slc/distTree.cpp \
#          ./slc/setup.cpp \
#          ./slc/slcConvert.cpp \
#          ./slc/slcSurface.cpp \
#          ./slc/surface.cpp \
          ./system/SetDirectory.cpp \
          ./system/thread/AutomateUtils.cpp \
          ./system/thread/CTime.cpp \
          ./system/thread/debug.cpp \
          ./system/thread/Edge.cpp \
          ./system/thread/EdgeGFX.cpp \
          ./system/thread/ExecBus.cpp \
          ./system/thread/NodeGFX.cpp \
          ./system/thread/ObjSubAutomate.cpp \
          ./system/thread/ObjSubAutomateCondExec.cpp \
          ./system/thread/ObjSubAutomateNodeExec.cpp \
          ./system/thread/StateMachine.cpp \
          ./system/thread/ThreadSimulation.cpp \
          ./system/thread/utils.cpp \
          ./system/thread/video.cpp 
