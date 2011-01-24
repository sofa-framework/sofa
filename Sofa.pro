SOFA_DIR =.
TEMPLATE = subdirs

message( "PRE-CONFIG: " $${CONFIG})

include($${SOFA_DIR}/sofa.cfg) 

SUBDIRS += extlibs/newmat

contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV
#SUBDIRS += extlibs/SLC
} # END SOFA_DEV

SUBDIRS += extlibs/qwt-5.2.0/src

contains(DEFINES,SOFA_XML_PARSER_TINYXML){
  SUBDIRS += extlibs/tinyxml
}

contains(DEFINES,SOFA_HAVE_ARTRACK){
  SUBDIRS += extlibs/ARTrack
}

# FlowVR
	SUBDIRS += extlibs/miniFlowVR
contains(DEFINES,SOFA_HAVE_FLOWVR){
	SUBDIRS -= extlibs/miniFlowVR
}

#CSParse

contains(DEFINES,SOFA_HAVE_CSPARSE){
	SUBDIRS += extlibs/csparse
}

contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV

#METIS

contains(DEFINES,SOFA_EXTLIBS_METIS){
	SUBDIRS += extlibs/metis
}

#TAUCS

contains(DEFINES,SOFA_EXTLIBS_TAUCS){
	contains(DEFINES,SOFA_HAVE_CILK){
		SUBDIRS += extlibs/taucs
	} else {
		SUBDIRS += extlibs/taucs-svn
	}
}

#FFMPEG
contains(DEFINES,SOFA_EXTLIBS_FFMPEG){
	SUBDIRS += extlibs/ffmpeg
}

} # END SOFA_DEV

#DCCD
contains(DEFINES,SOFA_HAVE_DCCD){
	SUBDIRS += extlibs/self-ccd-1.0/self-ccd.pro
}

#QGLViewer

contains(DEFINES,SOFA_GUI_QGLVIEWER){
	SUBDIRS += extlibs/libQGLViewer-2.3.3/QGLViewer
}


contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV
#CUDPP
contains(DEFINES,SOFA_GPU_CUDA){
	contains(DEFINES,SOFA_GPU_CUDPP){
		SUBDIRS += extlibs/cudpp
	}
}
} # END SOFA_DEV

contains(DEFINES,SOFA_HAVE_COLLADADOM){
	SUBDIRS += extlibs/colladadom/dom/colladadom.pro
}

# PML
	SUBDIRS += extlibs/PML
	SUBDIRS += extlibs/LML
!contains(DEFINES,SOFA_PML){
	SUBDIRS -= extlibs/PML
	SUBDIRS -= extlibs/LML
}

#FISHPACK
contains(DEFINES,SOFA_HAVE_FISHPACK){
	SUBDIRS += extlibs/fftpack
	SUBDIRS += extlibs/fishpack
}

# MUPARSER
contains(DEFINES,MUPARSER){
	SUBDIRS += extlibs/muparser
}

#VRPN
contains(DEFINES,SOFA_HAVE_VRPN){
	contains(DEFINES,VRPN_USE_WIIUSE){
		SUBDIRS += extlibs/wiiuse
	}
	SUBDIRS += extlibs/VRPN
}

SUBDIRS += framework
SUBDIRS += modules
SUBDIRS += applications

# Print current config

message( "====== SOFA Build Configuration ======")

contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV
message( "==== UNSTABLE DEVELOPMENT VERSION ====")
} # END SOFA_DEV

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
	contains( CONFIGSTATIC, static) {
		message( "|  Mode: DEBUG with static compilation")
	}
	else {
	  message( "|  Mode: DEBUG")
	}
}
contains (CONFIGDEBUG, release) {
  contains (QMAKE_CXXFLAGS,-g) {
    message( "|  Mode: RELEASE with debug symbols")
  }
  else {
    contains (CONFIGDEBUG, profile) {
      message( "|  Mode: RELEASE with profiling")
    }
    else {
			contains (CONFIGSTATIC, static) {
	      message( "|  Mode: RELEASE with static compilation")
			}
			else {
				message( "|  Mode : RELEASE")
			}
    }
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

contains(DEFINES,SOFA_HAVE_BOOST) {
  message( "|  BOOST libraries: ENABLED")
}
else {
  message( "|  BOOST libraries: DISABLED")
}

contains(DEFINES,SOFA_XML_PARSER_TINYXML) {
  message( "|  TinyXML parser: ENABLED")
}
else {
  message( "|  TinyXML parser: DISABLED")
}

contains(DEFINES,SOFA_XML_PARSER_LIBXML) {
  message( "|  LibXML parser: ENABLED")
}
else {
  message( "|  LibXML parser: DISABLED")
}

contains(DEFINES,SOFA_HAVE_PNG) {
  message( "|  PNG support: ENABLED")
}
else {
  message( "|  PNG support: DISABLED")
}

contains(DEFINES,SOFA_HAVE_GLEW) {
  message( "|  OpenGL Extensions support using GLEW: ENABLED")
}
else {
  message( "|  OpenGL Extensions support using GLEW: DISABLED")
}

contains(DEFINES,SOFA_GPU_CUDA) {
  message( "|  GPU support using CUDA: ENABLED")
}
else {
  message( "|  GPU support using CUDA: DISABLED")
}
contains(DEFINES,SOFA_SMP) {
  message( "|   Sofa-Parallel: ENABLED ")
  message( "| KAAPI_DIR=$${KAAPI_DIR}")
}
else {
  message( "|  Sofa-Parallel: DISABLED")
}

contains(DEFINES,SOFA_GPU_OPENCL) {
  message( "|  GPU support using OPENCL: ENABLED")
}
else {
  message( "|  GPU support using OPENCL: DISABLED")
}

contains(DEFINES,SOFA_PML) {
  message( "|  PML/LML support: ENABLED")
}
else {
  message( "|  PML/LML support: DISABLED")
}


contains(DEFINES,SOFA_HAVE_CSPARSE) {
  message( "|  CSPARSE library : ENABLED")
}
else {
  message( "|  CSPARSE library : DISABLED")
}

contains(DEFINES,SOFA_HAVE_METIS) {
  message( "|  METIS library : ENABLED")
}
else {
  message( "|  METIS library : DISABLED")
}

contains(DEFINES,SOFA_HAVE_TAUCS) {
  message( "|  TAUCS library : ENABLED")
contains(DEFINES,SOFA_HAVE_CILK) {
  message( "|  CILK library : ENABLED")
} else {
  message( "|  CILK library : DISABLE")
}
}
else {
  message( "|  TAUCS library : DISABLED")
}


contains(DEFINES,SOFA_GUI_GLUT) {
  message( "|  GLUT GUI: ENABLED")
}
else {
  message( "|  GLUT GUI: DISABLED")
}

contains(DEFINES,SOFA_DEV){ # BEGIN SOFA_DEV
contains(DEFINES,SOFA_GUI_FLTK) {
  message( "|  FLTK GUI: ENABLED")
}
else {
  message( "|  FLTK GUI: DISABLED")
}
} # END SOFA_DEV

!contains(DEFINES,SOFA_GUI_QTVIEWER) {
!contains(DEFINES,SOFA_GUI_QGLVIEWER) {
{
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

message( "======================================")
message( "|  CONFIG: " $${CONFIG})
message( "|  DEFINES: " $${DEFINES})
message( "======================================")



unix {
  contains(DEFINES, SOFA_QT4):DOLLAR="\\$"
  !contains(DEFINES, SOFA_QT4):DOLLAR="\$"
  contains (DEFINES, SOFA_SMP) {
    system(echo "export SOFA_DIR=$${PWD}" >config-Sofa-parallel.sh) 
    system(echo "export KAAPI_DIR=$${KAAPI_DIR}" >>config-Sofa-parallel.sh) 
    system(echo "export LD_LIBRARY_PATH=$${DOLLAR}SOFA_DIR/lib/linux:$${DOLLAR}KAAPI_DIR/lib:$${DOLLAR}LD_LIBRARY_PATH" >>config-Sofa-parallel.sh) 
    system(echo "export PATH=$${DOLLAR}SOFA_DIR/bin:$${DOLLAR}KAAPI_DIR/bin:$${DOLLAR}PATH" >>config-Sofa-parallel.sh) 
    contains (DEFINES, SOFA_GPU_CUDA) {
      system(echo "export CUDA_DIR=$${CUDA_DIR}" >>config-Sofa-parallel.sh) 
      system(echo "export LD_LIBRARY_PATH=$${DOLLAR}CUDA_DIR/lib:$${DOLLAR}CUDA_DIR/lib64:$${DOLLAR}LD_LIBRARY_PATH" >>config-Sofa-parallel.sh) 
      system(echo "export PATH=$${DOLLAR}CUDA_DIR/bin:$${DOLLAR}PATH" >>config-Sofa-parallel.sh) 
    }
  }
}
