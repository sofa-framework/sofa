cmake_minimum_required(VERSION 2.8)

include(CMakeDependentOption)

if(NOT GENERATED_FROM_MAIN_SOLUTION) # configuring from sub-project
	# load cache
	load_cache("${SOFA_ROOT_DIR}" INCLUDE_INTERNALS SOFA_SRC_DIR SOFA_BUILD_DIR)
else() # configuring from main solution
	include(${CMAKE_CURRENT_LIST_DIR}/configuration.cmake)
	
	# hide unused default cmake variables
	set(CMAKE_INSTALL_PREFIX "${SOFA_BUILD_DIR}" CACHE INTERNAL "Sofa install path (not used yet)")
	
	set(compilerDefines)

	# plugins (auto-search)
	file(GLOB pluginPathes "${SOFA_APPLICATIONS_PLUGINS_DIR}/*")
	foreach(pluginPath ${pluginPathes})
		if(IS_DIRECTORY ${pluginPath} AND EXISTS ${pluginPath}/CMakeLists.txt)
			file(RELATIVE_PATH pluginOriginalName ${SOFA_APPLICATIONS_PLUGINS_DIR} ${pluginPath})
			string(TOUPPER ${pluginOriginalName} pluginName)
			option("PLUGIN_${pluginName}" "Enable plugin ${pluginOriginalName}" OFF)
			RegisterDependencies(${pluginOriginalName} OPTION "PLUGIN_${pluginName}" COMPILE_DEFINITIONS "SOFA_HAVE_PLUGIN_${pluginName}" PATH "${pluginPath}")
		endif()
	endforeach()

	# dev-plugins (auto-search)
	file(GLOB devPluginPathes "${SOFA_APPLICATIONS_DEV_PLUGINS_DIR}/*")
	foreach(devPluginPath ${devPluginPathes})
		if(IS_DIRECTORY ${devPluginPath} AND EXISTS ${devPluginPath}/CMakeLists.txt)
			file(RELATIVE_PATH devPluginOriginalName ${SOFA_APPLICATIONS_DEV_PLUGINS_DIR} ${devPluginPath})
			string(TOUPPER ${devPluginOriginalName} devPluginName)
			option("PLUGIN-DEV_${devPluginName}" "Enable dev plugin ${devPluginOriginalName}" OFF)
			RegisterDependencies(${devPluginOriginalName} OPTION "PLUGIN-DEV_${devPluginName}" COMPILE_DEFINITIONS "SOFA_HAVE_DEVPLUGIN_${devPluginName}" PATH "${devPluginPath}")
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
	set(EXTERNAL_QT_PATH "${QTDIR}" CACHE PATH "Qt dir path")
	option(EXTERNAL_USE_QT4 "Use QT4 (else Sofa will use QT3) if Qt is needed" ON)
	if(EXTERNAL_USE_QT4)
		list(APPEND compilerDefines SOFA_QT4)
	endif()

	## boost
	option(EXTERNAL_HAVE_BOOST "Use the system boost library instead of extlib/miniBoost" OFF)
	if(EXTERNAL_HAVE_BOOST)
		set(EXTERNAL_BOOST_PATH "" CACHE PATH "Boost full version path (must contain the compiled libraries)")
		list(APPEND compilerDefines SOFA_HAVE_BOOST)
	endif()

	## zlib
	option(EXTERNAL_HAVE_ZLIB "Use the ZLib library" OFF)
	if(EXTERNAL_HAVE_ZLIB)
		list(APPEND compilerDefines SOFA_HAVE_ZLIB)
	endif()

	## libpng
	option(EXTERNAL_HAVE_PNG "Use the LibPNG library" OFF)
	if(EXTERNAL_HAVE_PNG)
		list(APPEND compilerDefines SOFA_HAVE_PNG)
	endif()

	## glew
	option(EXTERNAL_HAVE_GLEW "Use the GLEW library" OFF)
	if(EXTERNAL_HAVE_GLEW)
		list(APPEND compilerDefines SOFA_HAVE_GLEW)
	endif()

	## ffmpeg
	option(EXTERNAL_HAVE_FFMPEG "Use the FFMPEG library" OFF)
	if(EXTERNAL_HAVE_FFMPEG)
		list(APPEND compilerDefines SOFA_HAVE_FFMPEG)
	endif()

	## METIS
	option(EXTERNAL_HAVE_METIS "Use Metis" OFF)
	if(EXTERNAL_HAVE_METIS)
		list(APPEND compilerDefines SOFA_HAVE_METIS)
	endif()

	## CSPARSE
	option(EXTERNAL_HAVE_CSPARSE "Use CSparse" OFF)
	if(EXTERNAL_HAVE_CSPARSE)
		list(APPEND compilerDefines SOFA_HAVE_CSPARSE)
	endif()

	## FLOWVR
	option(EXTERNAL_HAVE_FLOWVR "Use FlowVR (otherwise miniFlowVR will be used from extlib)" OFF)
	if(EXTERNAL_HAVE_FLOWVR)
		list(APPEND compilerDefines SOFA_HAVE_FLOWVR)
	#TODO port features/sofa/flowvr.prf
	else()
		list(APPEND compilerDefines MINI_FLOWVR)
	endif()

	## EIGEN
	option(EXTERNAL_HAVE_EIGEN2 "Use Eigen" ON)
	if(EXTERNAL_HAVE_EIGEN2)
		list(APPEND compilerDefines SOFA_HAVE_EIGEN2)
	endif()

	# Optionnal features

	## PARDISO
	option(OPTION_PARDISO "Use Pardiso" OFF)
	if(OPTION_PARDISO)
		list(APPEND compilerDefines SOFA_HAVE_PARDISO)
	endif()

	## NO OPENGL
	option(OPTION_NO_OPENGL "Disable OpenGL" OFF)
	if(OPTION_NO_OPENGL)
		list(APPEND compilerDefines SOFA_NO_OPENGL)
		if (EXTERNAL_HAVE_GLEW)
			list(REMOVE_ITEM compilerDefines SOFA_HAVE_GLEW)
		endif()
		set(SOFA_VISUAL_LIB SofaBaseVisual)
	else()
		set(SOFA_VISUAL_LIB SofaOpenglVisual)
	endif()

	## NO QT
	option(OPTION_NO_QT "Disable QT" OFF)
	if(OPTION_NO_QT)
		list(APPEND compilerDefines SOFA_NO_QT)
	endif()
  
	## Tutorials
	option(OPTION_TUTORIALS "Build SOFA tutorials" OFF)
	if(OPTION_TUTORIALS)
		list(APPEND compilerDefines SOFA_HAVE_TUTORIALS)
	endif()
	
	## Applications
	option(OPTION_APPLICATIONS "Build SOFA applications (the various tools and editors using the libraries)" OFF)

	## PML
	option(OPTION_PML "PML support" OFF)
	if(OPTION_PML)
		list(APPEND compilerDefines SOFA_HAVE_PML)
	endif()

	## GPU OpenCL
	option(OPTION_GPU_OPENCL "OpenCL GPU support" OFF)
	if(OPTION_GPU_OPENCL)
		list(APPEND compilerDefines SOFA_GPU_OPENCL)
	endif()

	## XML
	option(XML_PARSER_LIBXML "Use LibXML instead of built-in TinyXML" OFF)
	if(XML_PARSER_LIBXML)
		list(APPEND compilerDefines SOFA_XML_PARSER_LIBXML)
	else()
		list(APPEND compilerDefines SOFA_XML_PARSER_TINYXML)
		list(APPEND compilerDefines TIXML_USE_STL)
		include_directories("${SOFA_EXTLIBS_DIR}/tinyxml")
	endif()

	# developer convenience
	#option(CONVENIENCE_ "" ON)

	# optionnal features
	option(SIMULATION_GRAPH_DAG "Directed acyclic graph" OFF)
	if(SIMULATION_GRAPH_DAG)
		list(APPEND compilerDefines SOFA_HAVE_DAG)
	endif()
	RegisterDependencies(SofaSimulationGraph OPTION "SIMULATION_GRAPH_DAG" COMPILE_DEFINITIONS "SOFA_HAVE_DAG" PATH "${SOFA_MODULES_DIR}/sofa/simulation/graph")
	
	option(SIMULATION_GRAPH_BGL "Boost graph library" OFF)
	if(SIMULATION_GRAPH_BGL)
		list(APPEND compilerDefines SOFA_HAVE_BGL)
	endif()
	RegisterDependencies(SofaSimulationBGL OPTION "SIMULATION_GRAPH_BGL" COMPILE_DEFINITIONS "SOFA_HAVE_BGL" PATH "${SOFA_MODULES_DIR}/sofa/simulation/bgl")

	CMAKE_DEPENDENT_OPTION(GUI_USE_QTVIEWER "Use QT Viewer" ON "NOT OPTION_NO_OPENGL;NOT OPTION_NO_QT" OFF)
	if(GUI_USE_QTVIEWER)
		list(APPEND compilerDefines SOFA_GUI_QTVIEWER)
	endif()
	CMAKE_DEPENDENT_OPTION(GUI_USE_QGLVIEWER "Use QGLViewer" OFF
		"NOT OPTION_NO_OPENGL; NOT OPTION_NO_QT" OFF)
	if(GUI_USE_QGLVIEWER)
		list(APPEND compilerDefines SOFA_GUI_QGLVIEWER)
	endif()
	CMAKE_DEPENDENT_OPTION(GUI_USE_GLUT "Use GLUT interface" ON
		"NOT OPTION_NO_OPENGL" OFF)
	if(GUI_USE_GLUT)
		list(APPEND compilerDefines SOFA_GUI_GLUT)
	endif()
	option(GUI_USE_INTERACTION "Enable interaction mode" OFF)
	if(GUI_USE_INTERACTION)
		list(APPEND compilerDefines SOFA_GUI_INTERACTION)
	endif()

	# unit tests
	option(UNIT-TESTS_USE "Build and use unit tests" OFF)
	if(UNIT-TESTS_USE)
		if(NOT WIN32)
                        option(UNIT-TESTS_BUILD_GTEST "Build google test framework" ON)
		endif()
	endif()

	# miscellaneous
	option(MISC_USE_DEVELOPER_MODE "Build and use the applications-dev projects (dev-plugins may need them)" OFF)
	if(MISC_USE_DEVELOPER_MODE)
		list(APPEND compilerDefines SOFA_DEV)
	endif()
	
	set(GLOBAL_COMPILER_DEFINES ${GLOBAL_COMPILER_DEFINES} ${compilerDefines} CACHE INTERNAL "Global Compiler Defines" FORCE)
	
	# os-specific
	if(XBOX)
		if(EXTERNAL_HAVE_BOOST)
			# we use EXTERNAL_BOOST_PATH but don't have the full boost and thus can't compile the code this normally enables.
			unset(EXTERNAL_HAVE_BOOST CACHE)
			list(REMOVE_ITEM compilerDefines SOFA_HAVE_BOOST)
		endif()
		if (EXTERNAL_HAVE_EIGEN2)
			# cpuid identification code does not exist on the platform, it's cleaner to disable it here.
			list(APPEND GLOBAL_COMPILER_DEFINES EIGEN_NO_CPUID)
		endif()
	endif()
endif()


