cmake_minimum_required(VERSION 2.8)

# hide unused default cmake variables
set(CMAKE_INSTALL_PREFIX "${SOFA_DIR}" CACHE INTERNAL "Sofa install path (not used yet)")

# plugins (auto-search)
file(GLOB pluginPathes "${SOFA_APPLICATIONS_PLUGINS_DIR}/*")
foreach(pluginPath ${pluginPathes})
        if(IS_DIRECTORY ${pluginPath} AND EXISTS ${pluginPath}/CMakeLists.txt)
		file(RELATIVE_PATH pluginName ${SOFA_APPLICATIONS_PLUGINS_DIR} ${pluginPath})
		string(TOUPPER ${pluginName} pluginName)
		set("SOFA_PLUGIN_PATH_${pluginName}" ${pluginPath} CACHE INTERNAL "Path to ${pluginName}")
		option("PLUGIN_${pluginName}" "Enable plugin ${pluginName}" OFF)
		set("SOFA_HAVE_PLUGIN_${pluginName}" "${PLUGIN_${pluginName}}")
		if("${SOFA_HAVE_PLUGIN_${pluginName}}")
			list(APPEND GLOBAL_COMPILER_DEFINES "SOFA_HAVE_PLUGIN_${pluginName}")
			list(APPEND SOFA_PLUGINS "SOFA_PLUGIN_PATH_${pluginName}")
		endif()
	endif()
endforeach()

# dev-plugins (auto-search)
file(GLOB devPluginPathes "${SOFA_APPLICATIONS_DEV_PLUGINS_DIR}/*")
foreach(devPluginPath ${devPluginPathes})
        if(IS_DIRECTORY ${devPluginPath} AND EXISTS ${devPluginPath}/CMakeLists.txt)
		file(RELATIVE_PATH devPluginName ${SOFA_APPLICATIONS_DEV_PLUGINS_DIR} ${devPluginPath})
		string(TOUPPER ${devPluginName} devPluginName)
		set("SOFA_DEV_PLUGIN_PATH_${devPluginName}" ${devPluginPath} CACHE INTERNAL "Path to ${devPluginName}")
		option("PLUGIN-DEV_${devPluginName}" "Enable dev plugin ${devPluginName}" OFF)
		set("SOFA_DEV_HAVE_PLUGIN_${devPluginName}" "${PLUGIN-DEV_${devPluginName}}")
		if("${SOFA_DEV_HAVE_PLUGIN_${devPluginName}}")
			list(APPEND GLOBAL_COMPILER_DEFINES "SOFA_DEV_HAVE_PLUGIN_${devPluginName}")
			list(APPEND SOFA_DEV_PLUGINS "SOFA_DEV_PLUGIN_PATH_${devPluginName}")
		endif()
	endif()
endforeach()

# extlibs

## qt
set(QTDIR $ENV{QTDIR})
if(NOT QTDIR STREQUAL "")
	file(GLOB QTDIR "${QTDIR}") # check if the QTDIR contains a correct path
endif()

if(WIN32)
	# convenience : try to find a valid QT path
	if(QTDIR STREQUAL "")
		set(QTDIR "${SOFA_TOOLS_DIR}/qt4win")
		file(GLOB QTDIR "${QTDIR}")
		if(QTDIR STREQUAL "")
			string(SUBSTRING "${SOFA_DIR}" 0 1 DISK_LETTER)
			file(GLOB QTDIRS "${DISK_LETTER}:/Qt*")
			list(GET QTDIRS 0 QTDIR)
		endif()
	endif()
endif()
# on Windows (and maybe also on Linux and Mac) the ENV{QTDIR} MUST BE DEFINED in order to find Qt (giving a path in find_package does not work)
set(EXTERNAL_QT_PATH "${QTDIR}" CACHE PATH "Qt dir path")
set(ENV{QTDIR} "${EXTERNAL_QT_PATH}")
option(EXTERNAL_USE_QT4 "Use QT4 (else Sofa will use QT3)" ON)

## boost
set(EXTERNAL_BOOST_PATH "" CACHE PATH "Use Boost full version (must contain the compiled libraries)")
if(EXTERNAL_BOOST_PATH STREQUAL "")
	unset(EXTERNAL_HAVE_BOOST CACHE)
else()
	set(EXTERNAL_HAVE_BOOST 1 CACHE INTERNAL "Use a full and compiled version of boost" FORCE)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_BOOST)
endif()

## zlib
option(EXTERNAL_HAVE_ZLIB "Use the ZLib library" ON)
if(EXTERNAL_HAVE_ZLIB)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_ZLIB)
endif()

## libpng
option(EXTERNAL_HAVE_PNG "Use the LibPNG library" ON)
if(EXTERNAL_HAVE_PNG)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_PNG)
endif()

## glew
option(EXTERNAL_HAVE_GLEW "Use the GLEW library" ON)
if(EXTERNAL_HAVE_GLEW)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_GLEW)
endif()

## glew
option(EXTERNAL_HAVE_FFMPEG "Use the FFMPEG library" OFF)
if(EXTERNAL_HAVE_FFMPEG)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_FFMPEG)
endif()

## CSPARSE
option(EXTERNAL_HAVE_CSPARSE "Use CSparse" ON)
if(EXTERNAL_HAVE_CSPARSE)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_CSPARSE)
endif()

## FLOWVR
option(EXTERNAL_HAVE_FLOWVR "Use FlowVR" ON)
if(EXTERNAL_HAVE_FLOWVR)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_FLOWVR)
	list(APPEND GLOBAL_COMPILER_DEFINES MINI_FLOWVR)
endif()

## EIGEN
option(EXTERNAL_HAVE_EIGEN2 "Use Eigen" ON)
if(EXTERNAL_HAVE_EIGEN2)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_EIGEN2)
endif()



# Optionnal features

## PARDISO
option(OPTION_PARDISO "Use Pardiso" OFF)
if(OPTION_PARDISO)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_PARDISO)
endif()

## NO OPENGL
option(OPTION_NO_OPENGL "Disable OpenGL" OFF)
if(OPTION_NO_OPENGL)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_NO_OPENGL)
endif()

## Tutorials
option(OPTION_TUTORIALS "Build SOFA tutorials" ON)
if(OPTION_TUTORIALS)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_TUTORIALS)
endif()



## XML
option(XML_PARSER_LIBXML "Use LibXML instead of built-in TinyXML" OFF)
if(XML_PARSER_LIBXML)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_XML_PARSER_LIBXML)
else()
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_XML_PARSER_TINYXML)
	list(APPEND GLOBAL_COMPILER_DEFINES TIXML_USE_STL)
	include_directories("${SOFA_EXTLIBS_DIR}/tinyxml")
endif()

# developer convenience
#option(CONVENIENCE_ "" ON)

# optionnal features
option(SIMULATION_GRAPH_DAG "Directed acyclic graph" ON)
if(SIMULATION_GRAPH_DAG)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_DAG)
endif()
option(SIMULATION_GRAPH_BGL "Boost graph library" OFF)
if(SIMULATION_GRAPH_BGL)
	list(APPEND GLOBAL_COMPILER_DEFINES SOFA_HAVE_BGL)
endif()

option(GUI_USE_QTVIEWER "Use QT Viewer" ON)
option(GUI_USE_QGLVIEWER "Use QT Viewer" ON)

# unit tests
option(UNIT-TESTS_USE "Build and use unit tests" ON)
if(NOT WIN32)
	option(UNIT-TESTS_BUILD_GTEST "Build google test framework" ON)
endif()

# miscellaneous
file(GLOB applicationDevExist "${SOFA_APPLICATIONS_DEV_DIR}")
if(applicationDevExist)
	set(MISC_USE_DEV_PROJECTS_MODE "ON")
else()
	set(MISC_USE_DEV_PROJECTS_MODE "OFF")
endif()
option(MISC_USE_DEV_PROJECTS "Build and use the applications-dev projects (dev-plugins may need them)" ${MISC_USE_DEV_PROJECTS_MODE})


