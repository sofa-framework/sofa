cmake_minimum_required(VERSION 2.8)

include(CMakeDependentOption)

if(NOT GENERATED_FROM_MAIN_SOLUTION) # configuring from sub-project
	# load cache
	load_cache("${SOFA_ROOT_DIR}" INCLUDE_INTERNALS SOFA_SRC_DIR SOFA_BUILD_DIR)
else() # configuring from main solution	
	# hide unused default cmake variables
	set(CMAKE_INSTALL_PREFIX "${SOFA_BUILD_DIR}" CACHE INTERNAL "Sofa install path (not used yet)")
	
	set(compilerDefines)

	# plugins (auto-search)
	file(GLOB pluginDirs "${SOFA_APPLICATIONS_PLUGINS_DIR}/*")
	foreach(pluginDir ${pluginDirs})
		if(IS_DIRECTORY ${pluginDir})
			get_filename_component(pluginName ${pluginDir} NAME)
			string(TOUPPER ${pluginName} pluginToUpperName)
			file(GLOB_RECURSE pluginPathes "${pluginDir}/*CMakeLists.txt") # WARNING: this wildcard expression can catch "example/badCMakeLists.txt"
			if(NOT pluginPathes STREQUAL "")
				option("SOFA-PLUGIN_${pluginToUpperName}" "Enable plugin ${pluginProjectName}" OFF)
				foreach(pluginPath ${pluginPathes})
					get_filename_component(projectFilename ${pluginPath} NAME)
					string(REPLACE "/${projectFilename}" "" pluginFolder ${pluginPath})
					get_filename_component(pluginProjectName ${pluginFolder} NAME)
					RegisterDependencies(${pluginProjectName} OPTION "SOFA-PLUGIN_${pluginToUpperName}" COMPILE_DEFINITIONS "SOFA_HAVE_PLUGIN_${pluginToUpperName}" PATH "${pluginFolder}")
				endforeach()
			endif()
		endif()
	endforeach()

	# dev-plugins (auto-search)
	file(GLOB devPluginDirs "${SOFA_APPLICATIONS_DEV_PLUGINS_DIR}/*")
	foreach(devPluginDir ${devPluginDirs})
		if(IS_DIRECTORY ${devPluginDir})
			get_filename_component(devPluginName ${devPluginDir} NAME)
			string(TOUPPER ${devPluginName} devPluginToUpperName)
			file(GLOB_RECURSE devPluginPathes "${devPluginDir}/*CMakeLists.txt") # WARNING: this wildcard expression can catch "example/badCMakeLists.txt"
			if(NOT devPluginPathes STREQUAL "")
				option("SOFA-DEVPLUGIN_${devPluginToUpperName}" "Enable dev plugin ${devPluginProjectName}" OFF)
				foreach(pluginPath ${pluginPathes})
					get_filename_component(devProjectFilename ${pluginPath} NAME)
					string(REPLACE "/${devProjectFilename}" "" devPluginFolder ${pluginPath})
					get_filename_component(devPluginProjectName ${devPluginFolder} NAME)
					RegisterDependencies(${devPluginProjectName} OPTION "SOFA-DEVPLUGIN_${devPluginToUpperName}" COMPILE_DEFINITIONS "SOFA_HAVE_DEVPLUGIN_${devPluginToUpperName}" PATH "${devPluginFolder}")
				endforeach()
			endif()
		endif()
	endforeach()

	# extlibs

	## qt
	set(QTDIR $ENV{QTDIR})
	if(NOT QTDIR STREQUAL "")
		if(WIN32)
			file(TO_CMAKE_PATH "${QTDIR}" QTDIR) # GLOB will fail with pathes containing backslashes.
		endif()
		file(GLOB QTDIR "${QTDIR}") # check if the QTDIR contains a correct path
	endif()

	### the ENV{QTDIR} MUST BE DEFINED in order to find Qt (giving a path in find_package does not work)
	set(SOFA-EXTERNAL_QT_PATH "${QTDIR}" CACHE PATH "Qt dir path")
	option(SOFA-EXTERNAL_PREFER_QT4 "Prefer Qt4 instead of Qt3 if Qt is needed" ON)
	if(SOFA-EXTERNAL_PREFER_QT4)
		list(APPEND compilerDefines SOFA_QT4)
	endif()

	## boost
	option(SOFA-EXTERNAL_HAVE_BOOST "Use the system boost library instead of extlib/miniBoost" OFF)
	if(SOFA-EXTERNAL_HAVE_BOOST)
		set(SOFA-EXTERNAL_BOOST_PATH "" CACHE PATH "Boost full version path (must contain the compiled libraries)")
		list(APPEND compilerDefines SOFA_HAVE_BOOST)
	endif()

	## zlib
	option(SOFA-EXTERNAL_HAVE_ZLIB "Use the ZLib library" OFF)
	if(SOFA-EXTERNAL_HAVE_ZLIB)
		list(APPEND compilerDefines SOFA_HAVE_ZLIB)
	endif()

	## libpng
	option(SOFA-EXTERNAL_HAVE_PNG "Use the LibPNG library" OFF)
	if(SOFA-EXTERNAL_HAVE_PNG)
		list(APPEND compilerDefines SOFA_HAVE_PNG)
	endif()

	## glew
	option(SOFA-EXTERNAL_HAVE_GLEW "Use the GLEW library" OFF)
	if(SOFA-EXTERNAL_HAVE_GLEW)
		list(APPEND compilerDefines SOFA_HAVE_GLEW)
	endif()

	## ffmpeg
	option(SOFA-EXTERNAL_HAVE_FFMPEG "Use the FFMPEG library" OFF)
	if(SOFA-EXTERNAL_HAVE_FFMPEG)
		list(APPEND compilerDefines SOFA_HAVE_FFMPEG)
	endif()

	## METIS
	option(SOFA-EXTERNAL_HAVE_METIS "Use Metis" OFF)
	if(SOFA-EXTERNAL_HAVE_METIS)
		list(APPEND compilerDefines SOFA_HAVE_METIS)
	endif()

	## CSPARSE
	option(SOFA-EXTERNAL_HAVE_CSPARSE "Use CSparse" OFF)
	option(SOFA-EXTERNAL_HAVE_FLOWVR "Use FlowVR (otherwise miniFlowVR will be used from extlib)" OFF) #TODO port features/sofa/flowvr.prf
	option(SOFA-EXTERNAL_HAVE_EIGEN2 "Use Eigen" ON)

	# Miscellaneous features

	## no opengl
	option(SOFA-MISC_NO_OPENGL "Disable OpenGL" OFF)
	if(SOFA-MISC_NO_OPENGL)
		list(APPEND compilerDefines SOFA_NO_OPENGL)
		if (SOFA-EXTERNAL_HAVE_GLEW)
			list(REMOVE_ITEM compilerDefines SOFA_HAVE_GLEW)
		endif()
		set(SOFA_VISUAL_LIB SofaBaseVisual)
	else()
		set(SOFA_VISUAL_LIB SofaOpenglVisual)
	endif()
	
	## application
	option(SOFA-APPLICATION_GENERATE_DOC "Build GenerateCoc application " OFF)
	option(SOFA-APPLICATION_GENERATE_RIGID "Build GenerateRigid application " OFF)
	option(SOFA-APPLICATION_GENERATE_TYPEDEFS "Build GenerateTypedefs application " OFF)
	option(SOFA-APPLICATION_MESH_CONV "Build MeshConv application " OFF)
	option(SOFA-APPLICATION_RUN_SOFA "Build RunSofa application " ON)
	option(SOFA-APPLICATION_SOFA_BATCH "Build SofaBatch application " ON)
	#option(SOFA-APPLICATION_SOFA_CONFIGURATION "Build SofaConfiguration application " OFF)
	option(SOFA-APPLICATION_MODELER "Build Modeler application " ON)
	option(SOFA-APPLICATION_SOFA_FLOWVR "Build SofaFlowVR application " OFF)
	option(SOFA-APPLICATION_SOFA_INFO "Build SofaInfo application " OFF)
	option(SOFA-APPLICATION_SOFA_INIT_TIMER "Build SofaInitTimer application " OFF)
	option(SOFA-APPLICATION_SOFA_OPENCL "Build SofaOpenCL application " OFF)
	option(SOFA-APPLICATION_SOFA_VERIFICATION "Build SofaVerification application " OFF)
  
	## tutorial
	option(SOFA-TUTORIAL_CHAIN_HYBRID "Build Chain hybrid tutorial" ON)
	option(SOFA-TUTORIAL_COMPOSITE_OBJECT "Build Composite object tutorial" ON)
	option(SOFA-TUTORIAL_HOUSE_OF_CARDS "Build House of cards tutorial" ON)
	option(SOFA-TUTORIAL_MIXED_PENDULUM "Build Mixed Pendulum tutorial" ON)
	option(SOFA-TUTORIAL_OBJECT_CREATOR "Build Object creator tutorial" OFF)
	option(SOFA-TUTORIAL_ONE_PARTICLE "Build One particle tutorial" ON)
	#option(SOFA-TUTORIAL_ONE_PARTICLE_WITH_SOFA_TYPEDEFS "Build One particle with sofa typedefs tutorial" OFF)
	option(SOFA-TUTORIAL_ONE_TETRAHEDRON "Build One tetrahedron tutorial" ON)
	#option(SOFA-TUTORIAL_ANATOMY_MODELLING "Build Anatomy modelling tutorial" OFF)

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
	option(SOFA-LIB_COMPONENT_EIGEN2_SOLVER "" OFF)

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
	mark_as_advanced(SOFA-LIB_COMPONENT_PARDISO_SOLVER)
	mark_as_advanced(SOFA-LIB_COMPONENT_RIGID)
	mark_as_advanced(SOFA-LIB_COMPONENT_SIMPLE_FEM)
	mark_as_advanced(SOFA-LIB_COMPONENT_SPARSE_SOLVER)

	mark_as_advanced(SOFA-LIB_PRECONDITIONER)
	mark_as_advanced(SOFA-LIB_SPH_FLUID)
	mark_as_advanced(SOFA-LIB_TAUCS_SOLVER)
	mark_as_advanced(SOFA-LIB_TOPOLOGY_MAPPING)
	mark_as_advanced(SOFA-LIB_USER_INTERACTION)
	mark_as_advanced(SOFA-LIB_VALIDATION)
	mark_as_advanced(SOFA-LIB_VOLUMETRIC_DATA)
	
	mark_as_advanced(SOFA-LIB_COMPONENT_SOFA_PML)

	mark_as_advanced(SOFA-LIB_COMPONENT_GPU_OPENCL)
endif()
	
	# simulation
	option(SOFA-LIB_SIMULATION_GRAPH_DAG "Directed acyclic graph" OFF)
	option(SOFA-LIB_SIMULATION_GRAPH_BGL "Boost graph library" OFF)

	# developer convenience
	#option(CONVENIENCE_ "" ON)

	# optionnal features
	CMAKE_DEPENDENT_OPTION(SOFA-LIB_GUI_QTVIEWER "Use QT Viewer" ON "NOT OPTION_NO_OPENGL;NOT OPTION_NO_QT" OFF)
	CMAKE_DEPENDENT_OPTION(SOFA-LIB_GUI_QGLVIEWER "Use QGLViewer" OFF
		"NOT SOFA-MISC_NO_OPENGL; NOT SOFA-MISC_NO_QT" OFF)
	CMAKE_DEPENDENT_OPTION(SOFA-LIB_GUI_GLUT "Use GLUT interface" ON
		"NOT SOFA-MISC_NO_OPENGL" OFF)
	option(SOFA-LIB_GUI_INTERACTION "Enable interaction mode" OFF)

	# unit tests
	option(SOFA-MISC_TESTS "Build and use unit tests" OFF)
	if(SOFA-MISC_TESTS)
		if(NOT WIN32)
			option(SOFA-MISC_BUILD_GTEST "Build google test framework" ON)
		endif()
	endif()

	# miscellaneous
	option(SOFA-MISC_DEVELOPER_MODE "Build and use the applications-dev projects (dev-plugins may need them)" OFF)
	if(SOFA-MISC_DEVELOPER_MODE)
		list(APPEND compilerDefines SOFA_DEV)
	endif()
	
	set(GLOBAL_COMPILER_DEFINES ${GLOBAL_COMPILER_DEFINES} ${compilerDefines} CACHE INTERNAL "Global Compiler Defines" FORCE)
	
	# os-specific
	if(XBOX)
		if(SOFA-EXTERNAL_HAVE_BOOST)
			# we use SOFA-EXTERNAL_BOOST_PATH but don't have the full boost and thus can't compile the code this normally enables.
			unset(SOFA-EXTERNAL_HAVE_BOOST CACHE)
			list(REMOVE_ITEM compilerDefines SOFA_HAVE_BOOST)
		endif()
		if (SOFA-EXTERNAL_HAVE_EIGEN2)
			# cpuid identification code does not exist on the platform, it's cleaner to disable it here.
			list(APPEND GLOBAL_COMPILER_DEFINES EIGEN_NO_CPUID)
		endif()
	endif()
endif()


