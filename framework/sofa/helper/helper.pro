# Target is a library: sofahelper
SOFA_DIR = ../../..
TEMPLATE = lib
TARGET = sofahelper
include($${SOFA_DIR}/sofa.cfg)
CONFIG += $$CONFIGLIBRARIES
!contains(CONFIGSTATIC, static) {
	CONFIG -= staticlib
CONFIG += dll
}
LIBS += $$SOFA_EXT_LIBS
INCLUDEPATH += /usr/include/libxml2
DEFINES += SOFA_BUILD_HELPER

# Cross-dependecy to faulttype is forbidden as defaulttype depends on helper...
# LIBS += -lsofadefaulttype$$LIBSUFFIX
# Make sure there are no cross-dependencies
INCLUDEPATH -= $$SOFA_DIR/modules
INCLUDEPATH -= $$SOFA_DIR/applications
HEADERS += helper.h \
    ArgumentParser.h \
    BackTrace.h \
    fixed_array.h \
    Factory.h \
    Factory.inl \
    FnDispatcher.h \
    FnDispatcher.inl \
    gl/Axis.h \
    gl/BasicShapes.h \
    gl/Capture.h \
    gl/Color.h \
    gl/Cylinder.h \
    gl/glfont.h \
    gl/glText.inl \
    gl/glText.h \
    gl/RAII.h \
    gl/template.h \
    gl/Texture.h \
    gl/Trackball.h \
    gl/Transformation.h \
    io/Image.h \
    io/ImageBMP.h \
    io/ImagePNG.h \
    io/ImageRAW.h \
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
    map.h \
    list.h \
    MarchingCubeUtility.h \
    MemoryManager.h \
    ParticleMask.h \
    PolarDecompose.h \
    Quater.h \
    Quater.inl \
    rmath.h \
    RandomGenerator.h \
    set.h \
    SVector.h \
    system/config.h \
    system/gl.h \
    system/glu.h \
    system/glut.h \
    system/SetDirectory.h \
    system/FileRepository.h \
    system/thread/CTime.h \
    system/thread/debug.h \
    system/thread/TimeoutWatchdog.h \
    system/PipeProcess.h \
    system/SofaOStream.h \
    system/DynamicLibrary.h \
    system/PluginManager.h \
    TagFactory.h \
    accessor.h \
    vector.h \
    vector_device.h \
    polygon_cube_intersection/vec.h \
    polygon_cube_intersection/polygon_cube_intersection.h \
    proximity.h \
    SimpleTimer.h \
    AdvancedTimer.h \
    io/ImageDDS.h \
	OptionsGroup.h \ 
    Polynomial_LD.h \
    Polynomial_LD.inl
SOURCES += ArgumentParser.cpp \
    BackTrace.cpp \
    Factory.cpp \
    gl/Axis.cpp \
    gl/Capture.cpp \
    gl/Color.cpp \
    gl/Cylinder.cpp \
    gl/glfont.cpp \
    gl/glText.cpp \
    gl/Texture.cpp \
    gl/Trackball.cpp \
    gl/Transformation.cpp \
    io/Image.cpp \
    io/ImageBMP.cpp \
    io/ImagePNG.cpp \
    io/ImageRAW.cpp \
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
    MarchingCubeUtility.cpp \
    Quater.cpp \
    RandomGenerator.cpp \
    system/SetDirectory.cpp \
    system/FileRepository.cpp \
    system/thread/CTime.cpp \
    system/thread/debug.cpp \
    system/thread/TimeoutWatchdog.cpp \
    system/PipeProcess.cpp \
    system/SofaOStream.cpp \
    system/DynamicLibrary.cpp \
    system/PluginManager.cpp \
    TagFactory.cpp \
    polygon_cube_intersection/polygon_cube_intersection.cpp \
    polygon_cube_intersection/fast_polygon_cube_intersection.cpp \
    vector.cpp \
    proximity.cpp \
    AdvancedTimer.cpp \
    io/ImageDDS.cpp \
    OptionsGroup.cpp \
	Polynomial_LD.cpp
contains(DEFINES,SOFA_HAVE_GLEW) { 
    HEADERS += gl/FrameBufferObject.h \
        gl/GLSLShader.h
    SOURCES += gl/FrameBufferObject.cpp \
        gl/GLSLShader.cpp
}
contains(DEFINES,SOFA_DEV) { # BEGIN SOFA_DEV
    HEADERS += \
    	DualQuat.inl \
        DualQuat.h 

    SOURCES += \
    	DualQuat.cpp 
}

contains(DEFINES,SOFA_HAVE_FFMPEG) { # SOFA_HAVE_FFMPEG
    HEADERS += \
    	gl/VideoRecorder.h 

    SOURCES += \
    	gl/VideoRecorder.cpp 
}

QMAKE_LFLAGS *= -ldl
