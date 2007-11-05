# Target is a library:  sofahelper

SOFA_DIR = ../../..
TEMPLATE = lib
include($${SOFA_DIR}/sofa.cfg)

TARGET = sofahelper$$LIBSUFFIX
CONFIG += $$CONFIGLIBRARIES
LIBS += $$SOFA_EXT_LIBS

# Cross-dependecy to faulttype is forbidden as defaulttype depends on helper...
# LIBS += -lsofadefaulttype$$LIBSUFFIX

# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_DIR/modules
INCLUDEPATH -= $$SOFA_DIR/applications

HEADERS += \
          ArgumentParser.h \
          BackTrace.h \
          fixed_array.h \
          Factory.h \
          Factory.inl \
          FnDispatcher.h \
          FnDispatcher.inl \
          gl/Axis.h \
          gl/Cylinder.h \
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
	  io/bvh/BVHChannels.h \
	  io/bvh/BVHJoint.h \
	  io/bvh/BVHLoader.h \
	  io/bvh/BVHMotion.h \
	  io/bvh/BVHOffset.h \
          LCPcalc.h \
          LCPSolver.h \
          LCPSolver.inl \
          PolarDecompose.h \
          Quater.h \
          Quater.inl \
          rmath.h \
          static_assert.h \
          system/config.h \
          system/SetDirectory.h \
          system/FileRepository.h \
          system/thread/CTime.h \
          system/thread/debug.h \
          vector.h \
	  polygon_cube_intersection/vec.h \
	  polygon_cube_intersection/polygon_cube_intersection.h
          
SOURCES += \
          ArgumentParser.cpp \
          BackTrace.cpp \
          Factory.cpp \
          gl/Axis.cpp \
          gl/Cylinder.cpp \
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
	  io/bvh/BVHJoint.cpp \
	  io/bvh/BVHLoader.cpp \
	  io/bvh/BVHMotion.cpp \
          LCPcalc.cpp \
          Quater.cpp \
          system/SetDirectory.cpp \
          system/FileRepository.cpp \
          system/thread/CTime.cpp \
          system/thread/debug.cpp \
	  polygon_cube_intersection/polygon_cube_intersection.cpp \
	  polygon_cube_intersection/fast_polygon_cube_intersection.cpp 
