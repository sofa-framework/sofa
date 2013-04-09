cmake_minimum_required(VERSION 2.8)

# useful pathes
set(SOFA_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "Path to the Sofa cmake directory")
set(SOFA_SRC_DIR ${CMAKE_SOURCE_DIR} CACHE INTERNAL "Path to the Sofa source directory")
set(SOFA_BUILD_DIR ${CMAKE_BINARY_DIR} CACHE INTERNAL "Path to the Sofa build directory")

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

# clear cached variables that we regenerate each time
set(GLOBAL_COMPILER_DEFINES "")

## os-specific
if(WIN32)
	list(APPEND GLOBAL_COMPILER_DEFINES "UNICODE")
        set(SOFA_LIB_OS_DIR "${SOFA_SRC_DIR}/lib/win32/Common" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
endif()

# cached variables
set(GLOBAL_COMPILER_DEFINES ${GLOBAL_COMPILER_DEFINES} CACHE INTERNAL "Global Compiler Defines" FORCE)

# cmake modules path, for our FindXXX.cmake modules
#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${SOFA_CMAKE_DIR}) #moved to pre.cmake

# NDEBUG preprocessor macro
if(CMAKE_BUILD_TYPE MATCHES "Release")
    list(APPEND GLOBAL_COMPILER_DEFINES "NDEBUG")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
    list(APPEND GLOBAL_COMPILER_DEFINES "SOFA_DEBUG")
endif()
