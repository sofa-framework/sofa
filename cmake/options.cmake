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
			list(APPEND GLOBAL_DEFINES "SOFA_HAVE_PLUGIN_${pluginName}")
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
			list(APPEND GLOBAL_DEFINES "SOFA_DEV_HAVE_PLUGIN_${devPluginName}")
			list(APPEND SOFA_DEV_PLUGINS "SOFA_DEV_PLUGIN_PATH_${devPluginName}")
		endif()
	endif()
endforeach()

# extlibs

## boost (do not raise SOFA_HAVE_BOOST if we use our mini-boost version)
set(SOFA_MINI_BOOST "${SOFA_EXTLIBS_DIR}/miniBoost")
set(EXTERNAL_BOOST_PATH ${SOFA_MINI_BOOST} CACHE PATH "Use our minimal boost or type in the path of a full boost version")
set(SOFA_BOOST_PATH "${EXTERNAL_BOOST_PATH}")
if((NOT EXTERNAL_BOOST_PATH STREQUAL "") AND (NOT EXTERNAL_BOOST_PATH STREQUAL ${SOFA_MINI_BOOST}))
	set(SOFA_HAVE_BOOST 1)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_BOOST)
endif()


## zlib
option(EXTERNAL_HAVE_ZLIB "Use the ZLib library" ON)
if(EXTERNAL_HAVE_ZLIB)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_ZLIB)
endif()

## libpng
option(EXTERNAL_HAVE_PNG "Use the LibPNG library" ON)
if(EXTERNAL_HAVE_PNG)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_PNG)
endif()

## glew
option(EXTERNAL_HAVE_GLEW "Use the GLEW library" ON)
if(EXTERNAL_HAVE_GLEW)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_GLEW)
endif()

## XML
option(XML_PARSER_LIBXML "Use LibXML instead of built-in TinyXML" OFF)
if(XML_PARSER_LIBXML)
	set(SOFA_XML_PARSER_LIBXML 1)
	list(APPEND GLOBAL_DEFINES SOFA_XML_PARSER_LIBXML)
else()
	set(SOFA_XML_PARSER_TINYXML 1)
	list(APPEND GLOBAL_DEFINES SOFA_XML_PARSER_TINYXML)
endif()

# developer convenience
option(CONVENIENCE_SPLIT_HEADERS_AND_SOURCES "Split headers and sources in different filters (virtual folders)" ON)

set(SOFA_HEADER_FILE_FILTER_NAME "")
set(SOFA_CPP_FILE_FILTER_NAME "")
if(CONVENIENCE_SPLIT_HEADERS_AND_SOURCES)
	set(SOFA_HEADER_FILE_FILTER_NAME	"Header Files")
	set(SOFA_CPP_FILE_FILTER_NAME		"Source Files")
endif()

# optionnal features
option(SIMULATION_GRAPH_DAG "Directed acyclic graph" ON)
if(SIMULATION_GRAPH_DAG)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_DAG)
endif()
option(SIMULATION_GRAPH_BGL "Boost graph library" OFF)
if(SIMULATION_GRAPH_BGL)
	list(APPEND GLOBAL_DEFINES SOFA_HAVE_BGL)
endif()


# miscellaneous
file(GLOB applicationDevExist "${SOFA_APPLICATIONS_DEV_DIR}")
if(applicationDevExist)
	set(MISC_USE_DEV_PROJECTS_MODE "ON")
else()
	set(MISC_USE_DEV_PROJECTS_MODE "OFF")
endif()
option(MISC_USE_DEV_PROJECTS "Build and use the applications-dev projects (dev-plugins may need them)" ${MISC_USE_DEV_PROJECTS_MODE})

option(MISC_USE_UNIT_TESTS "Build and use unit tests" OFF)


