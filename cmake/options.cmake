include(CMakeDependentOption)

# hide unused default cmake variables
# CHG: disable as this breaks build of external applications
# set(CMAKE_INSTALL_PREFIX "${SOFA_BUILD_DIR}" CACHE INTERNAL "Sofa install path (not used yet)")
	
set(compilerDefines)

set(SOFA-EXTERNAL_INCLUDE_DIR ${SOFA-EXTERNAL_INCLUDE_DIR} CACHE PATH "Include path for pre-compiled dependencies outside of the Sofa directory")
set(SOFA-EXTERNAL_LIBRARY_DIR ${SOFA-EXTERNAL_LIBRARY_DIR} CACHE PATH "Library path for pre-compiled dependencies outside of the Sofa directory")

# extlibs

## eigen
set(SOFA-EXTERNAL_EIGEN_PATH "${SOFA_EXTLIBS_DIR}/eigen-3.2.0" CACHE PATH "Path to the eigen header-only library")

## qt
set(QTDIR "$ENV{QTDIR}")
if(WIN32 AND QTDIR STREQUAL "")
	set(QTDIR "${SOFA_TOOLS_DIR}/qt4win")
endif()
if(NOT QTDIR STREQUAL "")
	if(WIN32)
		file(TO_CMAKE_PATH "${QTDIR}" QTDIR) # GLOB will fail with pathes containing backslashes.
	endif()
	file(GLOB QTDIR "${QTDIR}") # check if the QTDIR contains a correct path
endif()

### the ENV{QTDIR} MUST BE DEFINED in order to find Qt (giving a path in find_package does not work)
set(SOFA-EXTERNAL_QT_PATH "${QTDIR}" CACHE PATH "Qt dir path")
list(APPEND compilerDefines SOFA_QT4)

## boost
set(MINIBOOST_PATH "${SOFA_EXTLIBS_DIR}/miniBoost")
set(SOFA-EXTERNAL_BOOST_PATH "${MINIBOOST_PATH}" CACHE PATH "Boost path, set to blank if you want to use the boost you built on your system or set a path if you want to use a compiled boost")
if(SOFA-EXTERNAL_BOOST_PATH STREQUAL "${MINIBOOST_PATH}")
    unset(SOFA-EXTERNAL_BOOST CACHE)
else()
	set(SOFA-EXTERNAL_BOOST 1 CACHE INTERNAL "Use the system / user compiled boost library instead of miniBoost" FORCE)
	list(APPEND compilerDefines SOFA_HAVE_BOOST)
endif()

## geometric tools
set(SOFA-EXTERNAL_GEOMETRIC_TOOLS_PATH "" CACHE PATH "Path to Geometric tools folder containing the cmake project")
if(SOFA-EXTERNAL_GEOMETRIC_TOOLS_PATH) # since the lib could be in the system path we cannot check the path with the EXISTS function
	set(SOFA-EXTERNAL_GEOMETRIC_TOOLS 1 CACHE INTERNAL "Build and use geometric tools" FORCE)
	# list(APPEND compilerDefines SOFA_HAVE_GEOMETRIC_TOOLS) # set this compiler defininition using RegisterProjectDependencies (to avoid the need of rebuilding everything if you change this option)
else()
	unset(SOFA-EXTERNAL_GEOMETRIC_TOOLS CACHE)
endif()

## tinyxml
set(TINYXML_PATH "${SOFA_EXTLIBS_DIR}/tinyxml")
set(SOFA-EXTERNAL_TINYXML_PATH "${TINYXML_PATH}" CACHE PATH "")
if(SOFA-EXTERNAL_TINYXML_PATH STREQUAL "${TINYXML_PATH}")
    unset(SOFA-EXTERNAL_TINYXML CACHE)
else()
	set(SOFA-EXTERNAL_TINYXML 1 CACHE INTERNAL "Use the system / user compiled tinyxml library instead of miniBoost" FORCE)
endif()

#set(SOFA-EXTERNAL_TINYXML_INCLUDE_DIR "" CACHE PATH "For pre-compiled tinyxml: library where headers are available")
#set(SOFA-EXTERNAL_TINYXML_LIBRARY "" CACHE PATH "For pre-compiled tinyxml: release-mode library name")
#set(SOFA-EXTERNAL_TINYXML_DEBUG_LIBRARY "" CACHE PATH "For pre-compiled tinyxml: debug-mode library name")
#mark_as_advanced(SOFA-EXTERNAL_TINYXML_INCLUDE_DIR)
#mark_as_advanced(SOFA-EXTERNAL_TINYXML_LIBRARY)
#mark_as_advanced(SOFA-EXTERNAL_TINYXML_DEBUG_LIBRARY)

## zlib
option(SOFA-EXTERNAL_ZLIB "Use the ZLib library" ON)

## libpng
option(SOFA-EXTERNAL_PNG "Use the LibPNG library" ON)

## freeglut
option(SOFA-EXTERNAL_FREEGLUT "Use the FreeGLUT library (instead of regular GLUT)" OFF)

## glew
option(SOFA-EXTERNAL_GLEW "Use the GLEW library" ON)

## ffmpeg
set(FFMPEG_PATH "")
option(SOFA-EXTERNAL_FFMPEG "Use the FFMPEG library" OFF)
set(SOFA-EXTERNAL_FFMPEG_PATH "${FFMPEG_PATH}" CACHE PATH "")
if(SOFA-EXTERNAL_FFMPEG)
	list(APPEND compilerDefines SOFA_HAVE_FFMPEG)
endif()

## METIS
option(SOFA-EXTERNAL_METIS "Use Metis" OFF)

## CSPARSE
set(CSPARSE_PATH "${SOFA_EXTLIBS_DIR}/csparse")
option(SOFA-EXTERNAL_CSPARSE "Use CSparse" OFF)
set(SOFA-EXTERNAL_CSPARSE_PATH "${CSPARSE_PATH}" CACHE PATH "")

## FLOWVR
set(FLOWVR_PATH "${SOFA_EXTLIBS_DIR}/miniFlowVR")
set(SOFA-EXTERNAL_FLOWVR_PATH "${FLOWVR_PATH}" CACHE PATH "")
if(NOT SOFA-EXTERNAL_FLOWVR_PATH STREQUAL ${FLOWVR_PATH})
	set(SOFA-EXTERNAL_FLOWVR 1 CACHE INTERNAL "Use FlowVR" FORCE)
else()
	unset(SOFA-EXTERNAL_FLOWVR CACHE)
endif()

## OPENCASCADE
set(OPENCASCADE_PATH "")
set(SOFA-EXTERNAL_OPENCASCADE_PATH "${OPENCASCADE_PATH}" CACHE PATH "OpenCascade Path")
file(TO_CMAKE_PATH "${SOFA-EXTERNAL_OPENCASCADE_PATH}" OPENCASCADE_PATH)
set(SOFA-EXTERNAL_OPENCASCADE_PATH "${OPENCASCADE_PATH}" CACHE PATH "OpenCascade Path" FORCE)

# Miscellaneous features

option(SOFA-MISC_CMAKE_VERBOSE OFF)

option(SOFA-MISC_SMP "Use SMP" OFF)

## no opengl
option(SOFA-MISC_NO_OPENGL "Disable OpenGL" OFF)
if(SOFA-MISC_NO_OPENGL)
	list(APPEND compilerDefines SOFA_NO_OPENGL)
	set(SOFA_VISUAL_LIB SofaBaseVisual)
else()
	set(SOFA_VISUAL_LIB SofaOpenglVisual)
endif()

## SOFA_NO_UPDATE_BBOX
option(SOFA-MISC_NO_UPDATE_BBOX "No BBOX update" OFF) # SOFA_NO_UPDATE_BBOX
if(SOFA-MISC_NO_UPDATE_BBOX)
	list(APPEND compilerDefines SOFA_NO_UPDATE_BBOX)
endif()



## application
option(SOFA-APPLICATION_GENERATEDOC "Build GenerateCoc application " OFF)
option(SOFA-APPLICATION_GENERATERIGID "Build GenerateRigid application " OFF)
option(SOFA-APPLICATION_GENERATETYPEDEFS "Build GenerateTypedefs application " OFF)
option(SOFA-APPLICATION_MESHCONV "Build MeshConv application " OFF)
if(PS3)
option(SOFA-APPLICATION_RUNSOFA "Build RunSofa application " OFF)
option(SOFA-APPLICATION_SOFABATCH "Build SofaBatch application " OFF)
option(SOFA-APPLICATION_MODELER "Build Modeler application " OFF)
else()
option(SOFA-APPLICATION_RUNSOFA "Build RunSofa application " ON)
option(SOFA-APPLICATION_SOFABATCH "Build SofaBatch application " ON)
option(SOFA-APPLICATION_MODELER "Build Modeler application " ON)
endif()

#option(SOFA-APPLICATION_SOFA_CONFIGURATION "Build SofaConfiguration application " OFF)

option(SOFA-APPLICATION_SOFA_FLOWVR "Build SofaFlowVR application " OFF)
option(SOFA-APPLICATION_SOFA_INFO "Build SofaInfo application " OFF)
option(SOFA-APPLICATION_SOFA_INIT_TIMER "Build SofaInitTimer application " OFF)
option(SOFA-APPLICATION_SOFA_OPENCL "Build SofaOpenCL application " OFF)
option(SOFA-APPLICATION_SOFA_TYPEDEFS "Build SofaTypedefs application " OFF)
option(SOFA-APPLICATION_SOFA_VERIFICATION "Build SofaVerification application " OFF)

## tutorial
if(PS3) 
     option(SOFA-TUTORIAL_CHAIN_HYBRID "Build Chain hybrid tutorial" OFF) 
     option(SOFA-TUTORIAL_COMPOSITE_OBJECT "Build Composite object tutorial" OFF) 
     option(SOFA-TUTORIAL_HOUSE_OF_CARDS "Build House of cards tutorial" OFF) 
     option(SOFA-TUTORIAL_MIXED_PENDULUM "Build Mixed Pendulum tutorial" OFF) 
     option(SOFA-TUTORIAL_ONE_PARTICLE "Build One particle tutorial" OFF) 
     #option(SOFA-TUTORIAL_ONE_PARTICLE_WITH_SOFA_TYPEDEFS "Build One particle with sofa typedefs tutorial" OFF) 
     option(SOFA-TUTORIAL_ONE_TETRAHEDRON "Build One tetrahedron tutorial" OFF) 
     #option(SOFA-TUTORIAL_ANATOMY_MODELLING "Build Anatomy modelling tutorial" OFF) 
else() 
	option(SOFA-TUTORIAL_CHAIN_HYBRID "Build Chain hybrid tutorial" ON)
	option(SOFA-TUTORIAL_COMPOSITE_OBJECT "Build Composite object tutorial" ON)
	option(SOFA-TUTORIAL_HOUSE_OF_CARDS "Build House of cards tutorial" OFF)
	option(SOFA-TUTORIAL_MIXED_PENDULUM "Build Mixed Pendulum tutorial" ON)
	option(SOFA-TUTORIAL_ONE_PARTICLE "Build One particle tutorial" ON)
	#option(SOFA-TUTORIAL_ONE_PARTICLE_WITH_SOFA_TYPEDEFS "Build One particle with sofa typedefs tutorial" OFF)
	option(SOFA-TUTORIAL_ONE_TETRAHEDRON "Build One tetrahedron tutorial" ON)
	#option(SOFA-TUTORIAL_ANATOMY_MODELLING "Build Anatomy modelling tutorial" OFF)
endif()

# core
option(SOFA-LIB_CORE "" ON)
option(SOFA-LIB_DEFAULTTYPE "" ON)
option(SOFA-LIB_HELPER "" ON)

# component
option(SOFA-LIB_COMPONENT_BASE_ANIMATION_LOOP "" ON)
option(SOFA-LIB_COMPONENT_BASE_COLLISION "" ON)
option(SOFA-LIB_COMPONENT_BASE_LINEAR_SOLVER "" ON)
option(SOFA-LIB_COMPONENT_BASE_MECHANICS "" ON)
option(SOFA-LIB_COMPONENT_BASE_TOPOLOGY "" ON)
option(SOFA-LIB_COMPONENT_BASE_VISUAL "" ON)
option(SOFA-LIB_COMPONENT_BOUNDARY_CONDITION "" ON)

option(SOFA-LIB_COMPONENT_COMPONENT_ADVANCED "" ON)
option(SOFA-LIB_COMPONENT_COMPONENT_COMMON "" ON)
option(SOFA-LIB_COMPONENT_COMPONENT_GENERAL "" ON)
option(SOFA-LIB_COMPONENT_COMPONENT_MISC "" ON)
option(SOFA-LIB_COMPONENT_COMPONENT_BASE "" ON)
option(SOFA-LIB_COMPONENT_COMPONENT_MAIN "" ON)

option(SOFA-LIB_COMPONENT_CONSTRAINT "" ON)
option(SOFA-LIB_COMPONENT_DEFORMABLE "" ON)
option(SOFA-LIB_COMPONENT_DENSE_SOLVER "" ON)
option(SOFA-LIB_COMPONENT_EIGEN2_SOLVER "" ON)

option(SOFA-LIB_COMPONENT_ENGINE "" ON)
option(SOFA-LIB_COMPONENT_EULERIAN_FLUID "" ON)
option(SOFA-LIB_COMPONENT_EXPLICIT_ODE_SOLVER "" ON)
option(SOFA-LIB_COMPONENT_EXPORTER "" ON)
option(SOFA-LIB_COMPONENT_GRAPH_COMPONENT "" ON)
option(SOFA-LIB_COMPONENT_HAPTICS "" ON)
option(SOFA-LIB_COMPONENT_IMPLICIT_ODE_SOLVER "" ON)
option(SOFA-LIB_COMPONENT_LOADER "" ON)
option(SOFA-LIB_COMPONENT_MESH_COLLISION "" ON)
option(SOFA-LIB_COMPONENT_MISC "" ON)
option(SOFA-LIB_COMPONENT_MISC_COLLISION "" ON)
option(SOFA-LIB_COMPONENT_MISC_ENGINE "" ON)
option(SOFA-LIB_COMPONENT_MISC_FEM "" ON)
option(SOFA-LIB_COMPONENT_MISC_FORCEFIELD "" ON)
option(SOFA-LIB_COMPONENT_MISC_MAPPING "" ON)
option(SOFA-LIB_COMPONENT_MISC_SOLVER "" ON)
option(SOFA-LIB_COMPONENT_MISC_TOPOLOGY "" ON)
option(SOFA-LIB_COMPONENT_NON_UNIFORM_FEM "" ON)
option(SOFA-LIB_COMPONENT_OBJECT_INTERACTION "" ON)
option(SOFA-LIB_COMPONENT_OPENGL_VISUAL "" ON)
option(SOFA-LIB_COMPONENT_PARDISO_SOLVER "" OFF)
option(SOFA-LIB_COMPONENT_RIGID "" ON)
option(SOFA-LIB_COMPONENT_SIMPLE_FEM "" ON)
option(SOFA-LIB_COMPONENT_SPARSE_SOLVER "" OFF)

option(SOFA-LIB_COMPONENT_PRECONDITIONER "" ON)
option(SOFA-LIB_COMPONENT_SPH_FLUID "" ON)
option(SOFA-LIB_COMPONENT_TAUCS_SOLVER "" OFF)
option(SOFA-LIB_COMPONENT_TOPOLOGY_MAPPING "" ON)
option(SOFA-LIB_COMPONENT_USER_INTERACTION "" ON)
option(SOFA-LIB_COMPONENT_VALIDATION "" ON)
option(SOFA-LIB_COMPONENT_VOLUMETRIC_DATA "" ON)

option(SOFA-LIB_COMPONENT_SOFA_PML "" OFF)

option(SOFA-LIB_COMPONENT_GPU_OPENCL "" OFF)	

# i don't know if we mark default components as advanced or not
# it would enhance readability but thinking to look for
# advanced options is not really obvious
if(false)
mark_as_advanced(SOFA-LIB_COMPONENT_BASE_ANIMATION_LOOP)
mark_as_advanced(SOFA-LIB_COMPONENT_BASE_COLLISION)
mark_as_advanced(SOFA-LIB_COMPONENT_BASE_LINEAR_SOLVER)
mark_as_advanced(SOFA-LIB_COMPONENT_BASE_MECHANICS)
mark_as_advanced(SOFA-LIB_COMPONENT_BASE_TOPOLOGY)
mark_as_advanced(SOFA-LIB_COMPONENT_BASE_VISUAL)
mark_as_advanced(SOFA-LIB_COMPONENT_BOUNDARY_CONDITION)

mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_ADVANCED)
mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_COMMON)
mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_GENERAL)
mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_MISC)
mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_BASE)
mark_as_advanced(SOFA-LIB_COMPONENT_COMPONENT_MAIN)

mark_as_advanced(SOFA-LIB_COMPONENT_CONSTRAINT)
mark_as_advanced(SOFA-LIB_COMPONENT_DEFORMABLE)
mark_as_advanced(SOFA-LIB_COMPONENT_DENSE_SOLVER)
mark_as_advanced(SOFA-LIB_COMPONENT_EIGEN2_SOLVER)

mark_as_advanced(SOFA-LIB_COMPONENT_ENGINE)
mark_as_advanced(SOFA-LIB_COMPONENT_EULERIAN_FLUID)
mark_as_advanced(SOFA-LIB_COMPONENT_EXPLICIT_ODE_SOLVER)
mark_as_advanced(SOFA-LIB_COMPONENT_EXPORTER)
mark_as_advanced(SOFA-LIB_COMPONENT_GRAPH_COMPONENT)
mark_as_advanced(SOFA-LIB_COMPONENT_HAPTICS)
mark_as_advanced(SOFA-LIB_COMPONENT_IMPLICIT_ODE_SOLVER)
mark_as_advanced(SOFA-LIB_COMPONENT_LOADER)
mark_as_advanced(SOFA-LIB_COMPONENT_MESH_COLLISION)
mark_as_advanced(SOFA-LIB_COMPONENT_MISC)
mark_as_advanced(SOFA-LIB_COMPONENT_MISC_COLLISION)
mark_as_advanced(SOFA-LIB_COMPONENT_MISC_ENGINE)
mark_as_advanced(SOFA-LIB_COMPONENT_MISC_FEM)
mark_as_advanced(SOFA-LIB_COMPONENT_MISC_FORCEFIELD)
mark_as_advanced(SOFA-LIB_COMPONENT_MISC_MAPPING)
mark_as_advanced(SOFA-LIB_COMPONENT_MISC_SOLVER)
mark_as_advanced(SOFA-LIB_COMPONENT_MISC_TOPOLOGY)
mark_as_advanced(SOFA-LIB_COMPONENT_NON_UNIFORM_FEM)
mark_as_advanced(SOFA-LIB_COMPONENT_OBJECT_INTERACTION)
mark_as_advanced(SOFA-LIB_COMPONENT_OPENGL_VISUAL)
#mark_as_advanced(SOFA-LIB_COMPONENT_PARDISO_SOLVER)
mark_as_advanced(SOFA-LIB_COMPONENT_RIGID)
mark_as_advanced(SOFA-LIB_COMPONENT_SIMPLE_FEM)
#mark_as_advanced(SOFA-LIB_COMPONENT_SPARSE_SOLVER)

mark_as_advanced(SOFA-LIB_PRECONDITIONER)
mark_as_advanced(SOFA-LIB_SPH_FLUID)
#mark_as_advanced(SOFA-LIB_TAUCS_SOLVER)
mark_as_advanced(SOFA-LIB_TOPOLOGY_MAPPING)
mark_as_advanced(SOFA-LIB_USER_INTERACTION)
mark_as_advanced(SOFA-LIB_VALIDATION)
mark_as_advanced(SOFA-LIB_VOLUMETRIC_DATA)

#mark_as_advanced(SOFA-LIB_COMPONENT_SOFA_PML)

#mark_as_advanced(SOFA-LIB_COMPONENT_GPU_OPENCL)
endif()

# simulation
option(SOFA-LIB_SIMULATION_GRAPH_DAG "Directed acyclic graph" ON)


# optionnal features
CMAKE_DEPENDENT_OPTION(SOFA-LIB_GUI_QT "Use QT interface" ON "NOT SOFA-MISC_NO_OPENGL;NOT SOFA-MISC_NO_QT;NOT PS3;" OFF)
MARK_AS_ADVANCED(FORCE SOFA-LIB_GUI_QT) # to hide it from the user
CMAKE_DEPENDENT_OPTION(SOFA-LIB_GUI_QTVIEWER "Use QT Viewer" ON "NOT SOFA-MISC_NO_OPENGL;NOT SOFA-MISC_NO_QT;NOT PS3;" OFF)
CMAKE_DEPENDENT_OPTION(SOFA-LIB_GUI_QGLVIEWER "Use QGLViewer" OFF "NOT SOFA-MISC_NO_OPENGL; NOT SOFA-MISC_NO_QT;" OFF)
CMAKE_DEPENDENT_OPTION(SOFA-LIB_GUI_GLUT "Use GLUT interface" ON "NOT SOFA-MISC_NO_OPENGL" OFF)
option(SOFA-LIB_GUI_INTERACTION "Enable interaction mode" OFF)



# unit tests
option(SOFA-MISC_TESTS "Build and use unit tests" OFF)
if(SOFA-MISC_TESTS)
	if(NOT WIN32)
		option(SOFA-MISC_BUILD_GTEST "Build google test framework" ON)
	endif()
endif()

# use external template
option(SOFA-MISC_EXTERN_TEMPLATE "Use extern template" ON)
if(NOT SOFA-MISC_EXTERN_TEMPLATE)
	list(APPEND compilerDefines SOFA_NO_EXTERN_TEMPLATE)
endif()

# float / double or both
option(SOFA-MISC_USE_FLOAT "Use single precision floating point (float)" OFF)
option(SOFA-MISC_USE_DOUBLE "Use double precision floating point (double)" OFF)
if(SOFA-MISC_USE_FLOAT AND NOT SOFA-MISC_USE_DOUBLE)
	list(APPEND compilerDefines SOFA_FLOAT)
elseif(SOFA-MISC_USE_DOUBLE AND NOT SOFA-MISC_USE_FLOAT)
	list(APPEND compilerDefines SOFA_DOUBLE)
endif()

# use OpenMP multithreading
option(SOFA-MISC_OPENMP "Use OpenMP multithreading" OFF)

# os-specific
if(XBOX)
	if(SOFA-EXTERNAL_BOOST)
		# we use SOFA-EXTERNAL_BOOST_PATH but don't have the full boost and thus can't compile the code this normally enables.
		unset(SOFA-EXTERNAL_BOOST CACHE)
		list(REMOVE_ITEM compilerDefines SOFA_HAVE_BOOST)
	endif()
	
	# eigen - cpuid identification code does not exist on the platform, it's cleaner to disable it here.
	list(APPEND compilerDefines EIGEN_NO_CPUID)
endif()





##############
#### CUDA ####
##############
option(SOFA-CUDA_VERBOSE_PTXAS "SOFA-CUDA_VERBOSE_PTXAS" OFF)	
if(SOFA-CUDA_VERBOSE_PTXAS)
	set(VERBOSE_PTXAS --ptxas-options=-v)
endif()

#Option to activate double-precision support in CUDA (requires GT200+ GPU and -arch sm_13 flag)
option(SOFA-CUDA_DOUBLE "SOFA-CUDA_DOUBLE" OFF)
if(SOFA-CUDA_DOUBLE)
	add_definitions("-DSOFA_GPU_CUDA_DOUBLE")
	AddCompilerDefinitions("SOFA_GPU_CUDA_DOUBLE")
endif()

#Option to use IEEE 754-compliant floating point operations
option(SOFA-CUDA_PRECISE "SOFA-CUDA_PRECISE" OFF)
if(SOFA-CUDA_PRECISE)
	add_definitions("-DSOFA_GPU_CUDA_PRECISE")
	AddCompilerDefinitions("SOFA_GPU_CUDA_PRECISE")
endif()

# Option to get double-precision for sqrt/div...
# (requires compute capability >= 2 and CUDA_VERSION > 3.0)
# (with SOFA_GPU_CUDA_PRECISE and SOFA_GPU_CUDA_DOUBLE you get IEEE 754-compliant floating point
#  operations for addition and multiplication only)
option(SOFA-CUDA_DOUBLE_PRECISE "SOFA-CUDA_DOUBLE_PRECISE" OFF)
if(SOFA-CUDA_DOUBLE_PRECISE)
	add_definitions("-DSOFA_GPU_CUDA_DOUBLE_PRECISE")
	AddCompilerDefinitions("SOFA_GPU_CUDA_DOUBLE_PRECISE")
endif()

# Option to activate cublas support in CUDA (requires SOFA_GPU_CUDA_DOUBLE)
option(SOFA-CUDA_CUBLAS "SOFA-CUDA_CUBLAS" OFF)
if(SOFA-CUDA_CUBLAS)
	add_definitions("-DSOFA_GPU_CUBLAS")
	AddCompilerDefinitions("SOFA_GPU_CUBLAS")
endif()

# Option to activate CUDPP (for RadixSort)
option(SOFA-CUDA_CUDPP "SOFA-CUDA_CUDPP" OFF)
if(SOFA-CUDA_CUDPP)
	add_definitions("-DSOFA_GPU_CUDPP")
	AddCompilerDefinitions("SOFA_GPU_CUDPP")
	if(SOFA-EXTERNAL_CUDPP)
		AddLinkerDependencies(cudpp)
	endif()
endif()

# Option to activate THRUST (for RadixSort)
# Note: THRUST is included in CUDA SDK 4.0+, it is recommended to use it if available
option(SOFA-CUDA_THRUST "SOFA-CUDA_THRUST" OFF)
if(SOFA-CUDA_THRUST)
	add_definitions("-DSOFA_GPU_THRUST")
	AddCompilerDefinitions("SOFA_GPU_THRUST")
	if(SOFA-EXTERNAL_THRUST)
		AddLinkerDependencies(thrust)
	endif()
endif()

#variable to choose to which architecture to compile
set(SOFA-CUDA_SM "20" CACHE FILEPATH "Which architecture to compile")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch sm_${SOFA-CUDA_SM} -Xcompiler -fPIC)

# TODO   activate it automatically
option(SOFA-CUDA_GREATER_THAN_GCC44 "SOFA-CUDA_GREATER_THAN_GCC44" OFF)
if(SOFA-CUDA_GREATER_THAN_GCC44)
      set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--compiler-options -fno-inline)
endif()








# plugins (auto-search)
set(SOFA_PROJECT_FOLDER "SofaPlugin")
RetrieveDependencies("${SOFA_APPLICATIONS_PLUGINS_DIR}" "SOFA-PLUGIN_" "Enable plugin" "SOFA_HAVE_PLUGIN_" RECURSIVE)

# dev-plugins (auto-search)
set(SOFA_PROJECT_FOLDER "SofaDevPlugin")
RetrieveDependencies("${SOFA_APPLICATIONS_DEV_PLUGINS_DIR}" "SOFA-DEVPLUGIN_" "Enable dev plugin" "SOFA_HAVE_DEVPLUGIN_" RECURSIVE)

# projects (auto-search)
set(SOFA_PROJECT_FOLDER "SofaApplication")
RetrieveDependencies("${SOFA_APPLICATIONS_PROJECTS_DIR}" "SOFA-APPLICATION_" "Enable application" "SOFA_HAVE_APPLICATION_")

# dev-projects (auto-search)
set(SOFA_PROJECT_FOLDER "SofaDevApplication")
RetrieveDependencies("${SOFA_APPLICATIONS_DEV_PROJECTS_DIR}" "SOFA-DEVAPPLICATION_" "Enable dev application" "SOFA_HAVE_DEVAPPLICATION_")

set(SOFA_PROJECT_FOLDER "")
# configurable paths to use pre-compiled dependencies outside of the Sofa directory

set(GLOBAL_COMPILER_DEFINES ${GLOBAL_COMPILER_DEFINES} ${compilerDefines} CACHE INTERNAL "Global Compiler Defines" FORCE)
