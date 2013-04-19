cmake_minimum_required(VERSION 2.8)

# configuration
option(CONFIGURATION_CUSTOM "Generate custom projects" OFF) # custom configuration (not defined yet)
option(CONFIGURATION_CORE "Generate only core projects" OFF) # framework projects
option(CONFIGURATION_MINIMAL "Generate core and modules" OFF) # framework + modules projects
option(CONFIGURATION_BATCH "Generate core, modules and batch applications (no gui)" OFF) # batch configuration (no gui, no opengl)
option(CONFIGURATION_APPLICATION "Generate core, modules and applications (runSofa and Modeler)" OFF) # framework + modules + projects (no unit tests)
option(CONFIGURATION_FULL "Generate core, modules, applications, tutorials and unit tests" ON) # framework + modules + projects + tutorials + unit tests

set(configurationNum 0)
if(CONFIGURATION_FULL)
	set(configurationMode "full")
	math(EXPR configurationNum "${configurationNum} + 1")
endif()
if(CONFIGURATION_APPLICATION)
	set(configurationMode "application")
	math(EXPR configurationNum "${configurationNum} + 1")
endif()
if(CONFIGURATION_BATCH)
	set(configurationMode "batch")
	math(EXPR configurationNum "${configurationNum} + 1")
endif()
if(CONFIGURATION_MINIMAL)
	set(configurationMode "minimal")
	math(EXPR configurationNum "${configurationNum} + 1")
endif()
if(CONFIGURATION_CORE)
	set(configurationMode "core")
	math(EXPR configurationNum "${configurationNum} + 1")
endif()
if(CONFIGURATION_CUSTOM)
	set(configurationMode "custom")
	math(EXPR configurationNum "${configurationNum} + 1")
endif()

if(configurationNum EQUAL 0)
	set(configurationMode "custom")
	math(EXPR configurationNum "${configurationNum} + 1")
endif()

if(NOT configurationNum EQUAL 1)
	message(FATAL_ERROR "\nYou cannot have two configurations, please choose one\n\n")
endif()

set(CONFIGURATION_MODE ${configurationMode} CACHE INTERNAL "Configuration" FORCE)

if(NOT CONFIGURATION_MODE STREQUAL "custom")
	get_property(variableDocumentation CACHE CONFIGURATION_CUSTOM PROPERTY HELPSTRING)
	set(CONFIGURATION_CUSTOM 0 CACHE BOOL "${variableDocumentation}" FORCE)
endif()

if(NOT CONFIGURATION_MODE STREQUAL "core")
	get_property(variableDocumentation CACHE CONFIGURATION_CORE PROPERTY HELPSTRING)
	set(CONFIGURATION_CORE 0 CACHE BOOL "${variableDocumentation}" FORCE)
endif()

if(NOT CONFIGURATION_MODE STREQUAL "minimal")
	get_property(variableDocumentation CACHE CONFIGURATION_MINIMAL PROPERTY HELPSTRING)
	set(CONFIGURATION_MINIMAL 0 CACHE BOOL "${variableDocumentation}" FORCE)
endif()

if(NOT CONFIGURATION_MODE STREQUAL "batch")
	get_property(variableDocumentation CACHE CONFIGURATION_BATCH PROPERTY HELPSTRING)
	set(CONFIGURATION_BATCH 0 CACHE BOOL "${variableDocumentation}" FORCE)
endif()

if(NOT CONFIGURATION_MODE STREQUAL "application")
	get_property(variableDocumentation CACHE CONFIGURATION_APPLICATION PROPERTY HELPSTRING)
	set(CONFIGURATION_APPLICATION 0 CACHE BOOL "${variableDocumentation}" FORCE)
endif()

if(NOT CONFIGURATION_MODE STREQUAL "full")
	get_property(variableDocumentation CACHE CONFIGURATION_FULL PROPERTY HELPSTRING)
	set(CONFIGURATION_FULL 0 CACHE BOOL "${variableDocumentation}" FORCE)
endif()

if(PRECONFIGURE_DONE)
	# enable projects according to the chosen configuration
	message(STATUS "Configuration mode is : ${CONFIGURATION_MODE}")

	if(CONFIGURATION_MODE STREQUAL "full")		
	
		set(OPTION_TUTORIALS 1 CACHE BOOL "Build SOFA tutorials" FORCE)
		message(STATUS " - Tutorials : Enabled")
		
		set(OPTION_APPLICATIONS 1 CACHE BOOL "Build SOFA applications (the various tools and editors using the libraries)" FORCE)
		message(STATUS " - Applications : Enabled")
		
		set(GUI_USE_QTVIEWER 1 CACHE BOOL "Use QT Viewer" FORCE)
		#set(GUI_USE_QGLVIEWER 1 CACHE BOOL "Use QGLViewer" FORCE)
		set(GUI_USE_GLUT 1 CACHE BOOL "Use GLUT interface" FORCE)
		
		set(OPTION_NO_OPENGL 0 CACHE BOOL "Disable OpenGL" FORCE)
		set(OPTION_NO_QT 0 CACHE BOOL "Disable QT" FORCE)
		
		set(SIMULATION_GRAPH_DAG 1 CACHE BOOL "Directed acyclic graph" FORCE)

		#...
		
	endif()

	if(CONFIGURATION_MODE STREQUAL "application")
		
		set(OPTION_APPLICATIONS 1 CACHE BOOL "Build SOFA applications (the various tools and editors using the libraries)" FORCE)
		message(STATUS " - Applications : Enabled")
		
		set(GUI_USE_QTVIEWER 1 CACHE BOOL "Use QT Viewer" FORCE)
		#set(GUI_USE_QGLVIEWER 1 CACHE BOOL "Use QGLViewer" FORCE)
		set(GUI_USE_GLUT 1 CACHE BOOL "Use GLUT interface" FORCE)
		
		set(OPTION_NO_OPENGL 0 CACHE BOOL "Disable OpenGL" FORCE)
		set(OPTION_NO_QT 0 CACHE BOOL "Disable QT" FORCE)
		
		set(SIMULATION_GRAPH_DAG 1 CACHE BOOL "Directed acyclic graph" FORCE)
		
		#...
		
	endif()

	if(CONFIGURATION_MODE STREQUAL "batch")
		
		set(OPTION_APPLICATIONS 1 CACHE BOOL "Build SOFA applications (the various tools and editors using the libraries)" FORCE)
		message(STATUS " - Batch applications : Enabled")
		
		set(GUI_USE_QTVIEWER 0 CACHE BOOL "Use QT Viewer" FORCE)
		set(GUI_USE_QGLVIEWER 0 CACHE BOOL "Use QGLViewer" FORCE)
		set(GUI_USE_GLUT 0 CACHE BOOL "Use GLUT interface" FORCE)
		
		set(OPTION_NO_OPENGL 1 CACHE BOOL "Disable OpenGL" FORCE)
		set(OPTION_NO_QT 1 CACHE BOOL "Disable QT" FORCE)
		
		set(EXTERNAL_HAVE_EIGEN2 0 CACHE BOOL "Use Eigen" FORCE)
		
		set(SIMULATION_GRAPH_DAG 1 CACHE BOOL "Directed acyclic graph" FORCE)
		
		#...
		
	endif()

	if(CONFIGURATION_MODE STREQUAL "minimal")
		
		set(GUI_USE_QTVIEWER 0 CACHE BOOL "Use QT Viewer" FORCE)
		set(GUI_USE_QGLVIEWER 0 CACHE BOOL "Use QGLViewer" FORCE)
		set(GUI_USE_GLUT 0 CACHE BOOL "Use GLUT interface" FORCE)
		
		set(OPTION_NO_OPENGL 1 CACHE BOOL "Disable OpenGL" FORCE)
		set(OPTION_NO_QT 1 CACHE BOOL "Disable QT" FORCE)
		
		set(EXTERNAL_HAVE_EIGEN2 0 CACHE BOOL "Use Eigen" FORCE)
		
		EnableProject("SofaComponentMain")
		#...
		
	endif()

	if(CONFIGURATION_MODE STREQUAL "core")
		
		set(GUI_USE_QTVIEWER 0 CACHE BOOL "Use QT Viewer" FORCE)
		set(GUI_USE_QGLVIEWER 0 CACHE BOOL "Use QGLViewer" FORCE)
		set(GUI_USE_GLUT 0 CACHE BOOL "Use GLUT interface" FORCE)
		
		set(OPTION_NO_OPENGL 1 CACHE BOOL "Disable OpenGL" FORCE)
		set(OPTION_NO_QT 1 CACHE BOOL "Disable QT" FORCE)
		
		set(EXTERNAL_HAVE_EIGEN2 0 CACHE BOOL "Use Eigen" FORCE)
		
		EnableProject("SofaCore")
		#..
		
	endif()

	message(STATUS "")
	
	# to allow options customization we must change the configuration to custom
	
	get_property(variableDocumentation CACHE CONFIGURATION_CUSTOM PROPERTY HELPSTRING)
	set(CONFIGURATION_CUSTOM 1 CACHE BOOL "${variableDocumentation}" FORCE)
	
	get_property(variableDocumentation CACHE CONFIGURATION_CORE PROPERTY HELPSTRING)
	set(CONFIGURATION_CORE 0 CACHE BOOL "${variableDocumentation}" FORCE)
	
	get_property(variableDocumentation CACHE CONFIGURATION_MINIMAL PROPERTY HELPSTRING)
	set(CONFIGURATION_MINIMAL 0 CACHE BOOL "${variableDocumentation}" FORCE)
	
	get_property(variableDocumentation CACHE CONFIGURATION_BATCH PROPERTY HELPSTRING)
	set(CONFIGURATION_BATCH 0 CACHE BOOL "${variableDocumentation}" FORCE)
	
	get_property(variableDocumentation CACHE CONFIGURATION_APPLICATION PROPERTY HELPSTRING)
	set(CONFIGURATION_APPLICATION 0 CACHE BOOL "${variableDocumentation}" FORCE)
	
	get_property(variableDocumentation CACHE CONFIGURATION_FULL PROPERTY HELPSTRING)
	set(CONFIGURATION_FULL 0 CACHE BOOL "${variableDocumentation}" FORCE)
	
	set(configurationMode "custom")
endif()