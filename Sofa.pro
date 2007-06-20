SOFA_DIR=.
TEMPLATE = subdirs

include($$SOFA_DIR/sofa.cfg)

SUBDIRS += extlibs/NewMAT
SUBDIRS += extlibs/SLC
# PML
contains(DEFINES,SOFA_PML){
	SUBDIRS += extlibs/PML
	SUBDIRS += extlibs/LML
}

SUBDIRS += framework
SUBDIRS += modules
SUBDIRS += applications

# Print current config

message( "====== SOFA Build Configuration ======")
win32 {
  message( "|  Platform: Windows")
}
else:macx {
  message( "|  Platform: MacOS")
}
else:unix {
  message( "|  Platform: Linux/Unix")
}

contains (CONFIGDEBUG, debug) {
  message( "|  Mode: DEBUG")
}
contains (CONFIGDEBUG, release) {
  contains (QMAKE_CXXFLAGS,-g) {
    message( "|  Mode: RELEASE with debug symbols")
  }
  else {
    message( "|  Mode: RELEASE")
  }
}

contains(DEFINES,SOFA_QT4) {
  message( "|  Qt version: 4.x")
}
else {
  message( "|  Qt version: 3.x")
}

contains(DEFINES,SOFA_RDTSC) {
  message( "|  RDTSC timer: ENABLED")
}
else {
  message( "|  RDTSC timer: DISABLED")
}

contains(DEFINES,SOFA_HAVE_PNG) {
  message( "|  PNG support: ENABLED")
}
else {
  message( "|  PNG support: DISABLED")
}

contains(DEFINES,SOFA_GPU_CUDA) {
  message( "|  GPU support using CUDA: ENABLED")
}
else {
  message( "|  GPU support using CUDA: DISABLED")
}

contains(DEFINES,SOFA_PML) {
  message( "|  PML/LML support: ENABLED")
}
else {
  message( "|  PML/LML support: DISABLED")
}

!contains(DEFINES,SOFA_GUI_QTVIEWER) {
!contains(DEFINES,SOFA_GUI_QGLVIEWER) {
!contains(DEFINES,SOFA_GUI_QTOGREVIEWER) {
  message( "|  Qt GUI: DISABLED")
}
else {
  message( "|  Qt GUI: ENABLED")
}
}
else {
  message( "|  Qt GUI: ENABLED")
}
}
else {
  message( "|  Qt GUI: ENABLED")
}

contains(DEFINES,SOFA_GUI_QTVIEWER) {
  message( "|  -  Qt OpenGL viewer: ENABLED")
}
else {
  message( "|  -  Qt OpenGL viewer: DISABLED")
}

contains(DEFINES,SOFA_GUI_QGLVIEWER) {
  message( "|  -  Qt QGLViewer viewer: ENABLED")
}
else {
  message( "|  -  Qt QGLViewer viewer: DISABLED")
}
contains(DEFINES,SOFA_GUI_QTOGREVIEWER) {
  message( "|  -  Qt OGRE 3D viewer: ENABLED")
}
else {
  message( "|  -  Qt OGRE 3D viewer: DISABLED")
}

contains(DEFINES,SOFA_GUI_FLTK) {
  message( "|  FLTK GUI: ENABLED")
}
else {
  message( "|  FLTK GUI: DISABLED")
}

message( "======================================")
