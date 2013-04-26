cmake_minimum_required(VERSION 2.8)

# cmake modules path, for our FindXXX.cmake modules
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${SOFA_CMAKE_DIR})

# useful pathes
set(SOFA_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "Path to the Sofa cmake directory")

get_filename_component(SOFA_ROOT_DIR ${SOFA_CMAKE_DIR} PATH)
set(SOFA_ROOT_DIR ${SOFA_ROOT_DIR} CACHE INTERNAL "Path to the Sofa root directory")

if(GENERATED_FROM_MAIN_SOLUTION)
	set(SOFA_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "Path to the Sofa source directory")
	set(SOFA_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "Path to the Sofa build directory")
	#message(STATUS "CMAKE_CURRENT_LIST_DIR = ${CMAKE_CURRENT_LIST_DIR}")
	#message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
	#message(STATUS "CMAKE_CURRENT_BINARY_DIR = ${CMAKE_CURRENT_BINARY_DIR}")
	#message(STATUS "SOFA_CMAKE_DIR = ${SOFA_CMAKE_DIR}")
	#message(STATUS "SOFA_ROOT_DIR = ${SOFA_ROOT_DIR}")
	#message(STATUS "SOFA_SRC_DIR = ${SOFA_SRC_DIR}")
	#message(STATUS "SOFA_BUILD_DIR = ${SOFA_BUILD_DIR}")
endif()

set(SOFA_BIN_DIR "${SOFA_BUILD_DIR}/bin" CACHE INTERNAL "Path to the Sofa bin directory")
if(WIN32)
    set(SOFA_INC_DIR "${SOFA_SRC_DIR}/include" CACHE INTERNAL "Path to the Sofa include directory")
endif()
set(SOFA_LIB_DIR "${SOFA_BUILD_DIR}/lib" CACHE INTERNAL "Path to the Sofa lib directory")
set(SOFA_EXTLIBS_DIR "${SOFA_SRC_DIR}/extlibs" CACHE INTERNAL "Path to the Sofa extlibs directory")
set(SOFA_SHARE_DIR "${SOFA_SRC_DIR}/share" CACHE INTERNAL "Path to the Sofa share directory")
set(SOFA_FRAMEWORK_DIR "${SOFA_SRC_DIR}/framework" CACHE INTERNAL "Path to the Sofa framework directory")
set(SOFA_MODULES_DIR "${SOFA_SRC_DIR}/modules" CACHE INTERNAL "Path to the Sofa modules directory")
set(SOFA_APPLICATIONS_DIR "${SOFA_SRC_DIR}/applications" CACHE INTERNAL "Path to the Sofa applications directory")
set(SOFA_APPLICATIONS_DEV_DIR "${SOFA_SRC_DIR}/applications-dev" CACHE INTERNAL "Path to the Sofa applications-dev directory")
set(SOFA_APPLICATIONS_PLUGINS_DIR "${SOFA_APPLICATIONS_DIR}/plugins" CACHE INTERNAL "Path to the Sofa applications plugins directory")
set(SOFA_APPLICATIONS_DEV_PLUGINS_DIR "${SOFA_APPLICATIONS_DEV_DIR}/plugins" CACHE INTERNAL "Path to the Sofa applications-dev plugin directory")
set(SOFA_TOOLS_DIR "${SOFA_SRC_DIR}/tools" CACHE INTERNAL "Path to the Sofa tools directory")

# useful settings
set(SOFA_VERSION_NUM "1_0" CACHE STRING "Version number for this build.")

## os-specific
if(WIN32)
	if(CMAKE_CL_64) 
		set(SOFA_LIB_OS_DIR "${SOFA_SRC_DIR}/lib/win64/Common" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
	else() 
		set(SOFA_LIB_OS_DIR "${SOFA_SRC_DIR}/lib/win32/Common" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
	endif() 
endif()
if(XBOX)
	set(SOFA_LIB_OS_DIR "${SOFA_SRC_DIR}/lib/xbox/Common" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
endif()

# cmake modules path, for our FindXXX.cmake modules
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${SOFA_CMAKE_DIR}) #moved to preProject.cmake

# disable every pre-enabled modules
foreach(dependency ${GLOBAL_DEPENDENCIES})
	unset(GLOBAL_PROJECT_ENABLED_${dependency} CACHE)
endforeach()

# clear cached variables that we regenerate each time
unset(GLOBAL_DEPENDENCIES CACHE) # reset the dependency database (used to compute interdependencies)
unset(GLOBAL_COMPILER_DEFINES CACHE)
unset(GLOBAL_INCLUDE_DIRECTORIES CACHE)