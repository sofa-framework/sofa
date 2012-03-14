load(sofa/pre)

TEMPLATE = subdirs

!contains(DEFINES, SOFA_DEV): message("WARNING: SOFA_DEV not defined, in-development code will be disabled!")

contains(DEFINES, SOFA_RELEASE): message("WARNING: SOFA_RELEASE defined, in-development code will be disabled!")

message( "PRE-CONFIG: " $${CONFIG})


########################################################################
# Enable plugins in addition of the standard Sofa libraries
########################################################################

usePlugin(PluginExample) 

contains(DEFINES, SOFA_HAVE_Compliant) { usePlugin(Compliant) }

contains(DEFINES, SOFA_HAVE_SENSABLE) {
        usePlugin(Sensable)
}

contains(DEFINES, SOFA_HAVE_PYTHON) {
        usePlugin(Python)
}

contains(DEFINES, SOFA_HAVE_ASYNCHROHAPTICS) {
	usePlugin(AsynchroHaptics)
}

contains(DEFINES,SOFA_HAVE_OPTITRACK) {
	usePlugin(OptiTrackNatNet)
}

!contains (DEFINES, SOFA_RELEASE) { # BEGIN !SOFA_RELEASE

contains (DEFINES, SOFA_HAVE_VRPN) {
	usePlugin(SofaVRPNClient)
}

contains(DEFINES, SOFA_HAVE_ARTRACK) {
        usePlugin(ARTrack)
}

contains(DEFINES, SOFA_HAVE_XITACT) {
	usePlugin(Xitact)
}

contains (DEFINES, SOFA_HAVE_HAPTION) {
	usePlugin(Haption)
}

contains (DEFINES, SOFA_HAVE_QTOGREVIEWER) {
    usePlugin(QtOgreViewer)
}

#usePlugin(PhysicsBasedInteractiveModeler)
#usePlugin(ldidetection)

} # END !SOFA_RELEASE

contains (DEFINES, SOFA_DEV) { # BEGIN SOFA_DEV 

	contains (DEFINES, SOFA_HAVE_VULCAIN) {
                usePlugin(vulcain)
	}

        contains (DEFINES, SOFA_HAVE_ldidetection) {
                usePlugin(ldidetection)
        }

        contains (DEFINES, SOFA_HAVE_LEM) {
		usePlugin(lem)
	}

	contains(DEFINES, SOFA_HAVE_TRIANGULARMESHREFINER) {
		usePlugin(TriangularMeshRefiner)
	}

	contains (DEFINES, SOFA_HAVE_BEAMADAPTER) {
		usePlugin(BeamAdapter)
	}

	contains (DEFINES, SOFA_HAVE_SHELL) {
                usePlugin(shells)
	}

        contains (DEFINES, SOFA_HAVE_CGAL) {
		usePlugin(CGALPlugin)
	}

	contains(DEFINES, SOFA_HAVE_FRAME) {
		usePlugin(frame)
	}
	
        contains(DEFINES, SOFA_HAVE_optixdetection) {
                usePlugin(optixdetection)
        }

        contains(DEFINES, SOFA_HAVE_IMAGE) {
		usePlugin(image)
	}	

	contains (DEFINES, SOFA_HAVE_REGISTRATION) {
		usePlugin(Registration)
	}

	contains (DEFINES, SOFA_HAVE_OPENCV) {
		usePlugin(OpenCVPlugin)
	}

	contains (DEFINES, SOFA_HAVE_PHYSICALFIELDMODELING) {
		usePlugin(PhysicalFieldModeling)
	}

	contains (DEFINES, SOFA_GPU_CUDA) { # BEGIN SOFA_GPU_CUDA

		contains (DEFINES, SOFA_HAVE_TRIANGULARMESHBASEDHEXASCUTTER) {
			usePlugin(TriangularMeshBasedHexasCutter)
		}

		contains (DEFINES, SOFA_HAVE_VOXELIZER) {
			usePlugin(Voxelizer)
		}

	} # END SOFA_GPU_CUDA

	contains (DEFINES, SOFA_HAVE_STEPLOADER) { # BEGIN SOFA_HAVE_STEPLOADER
		usePlugin(MeshSTEPLoader)
	}

	contains (DEFINES, SOFA_HAVE_PERSISTENTCONTACT) {
		usePlugin(PersistentContact)
	}

	contains(DEFINES, SOFA_HAVE_ASCLEPIOS) {
		usePlugin(sofa-asclepios)
	}

contains(DEFINES, SOFA_HAVE_PLUGIN_FEM) {
	usePlugin(FEM)
}

contains (DEFINES, SOFA_HAVE_SOHUSIM) {
	usePlugin(Sohusim)
}

contains (DEFINES, SOFA_HAVE_STABLEFLUID_PLUGIN) {
	usePlugin(StableFluidBehaviorPlugin)
	usePlugin(StableFluidModelPlugin)
}

contains (DEFINES, SOFA_HAVE_MANIFOLDTOPOLOGIES) {
	usePlugin(ManifoldTopologies)
}


} # END SOFA_DEV


########################################################################
# Generate SUBDIRS specifications to build everything
########################################################################

buildEnabledArtifacts()

########################################################################
# Print current config
########################################################################

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

contains(DEFINES,SOFA_HAVE_PYTHON) {
  message( "|  PYTHON script support: ENABLED")
}
else {
  message( "|  PYTHON script support: DISABLED")
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

!contains(DEFINES,SOFA_GUI_QTVIEWER) {
!contains(DEFINES,SOFA_GUI_QGLVIEWER) {
{
  message( "|  Qt GUI: DISABLED")
}
#else {
 # message( "|  Qt GUI: ENABLED")
#}
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

load(sofa/post)
