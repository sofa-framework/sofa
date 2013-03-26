cmake_minimum_required(VERSION 2.8)

# useful pathes
set(SOFA_CMAKE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE INTERNAL "Path to the Sofa cmake directory")
get_filename_component(SOFA_DIR ${SOFA_CMAKE_DIR} PATH)
set(SOFA_DIR ${SOFA_DIR} CACHE INTERNAL "Path to the Sofa root directory")
set(SOFA_BIN_DIR "${SOFA_DIR}/bin" CACHE INTERNAL "Path to the Sofa bin directory")
set(SOFA_INC_DIR "${SOFA_DIR}/include" CACHE INTERNAL "Path to the Sofa include directory")
set(SOFA_LIB_DIR "${SOFA_DIR}/lib" CACHE INTERNAL "Path to the Sofa lib directory")
set(SOFA_EXTLIBS_DIR "${SOFA_DIR}/extlibs" CACHE INTERNAL "Path to the Sofa extlibs directory")
set(SOFA_SHARE_DIR "${SOFA_DIR}/share" CACHE INTERNAL "Path to the Sofa share directory")
set(SOFA_FRAMEWORK_DIR "${SOFA_DIR}/framework" CACHE INTERNAL "Path to the Sofa framework directory")
set(SOFA_MODULES_DIR "${SOFA_DIR}/modules" CACHE INTERNAL "Path to the Sofa modules directory")
set(SOFA_APPLICATIONS_DIR "${SOFA_DIR}/applications" CACHE INTERNAL "Path to the Sofa applications directory")
set(SOFA_APPLICATIONS_DEV_DIR "${SOFA_DIR}/applications-dev" CACHE INTERNAL "Path to the Sofa applications-dev directory")
set(SOFA_APPLICATIONS_PLUGINS_DIR "${SOFA_APPLICATIONS_DIR}/plugins" CACHE INTERNAL "Path to the Sofa applications plugins directory")
set(SOFA_APPLICATIONS_DEV_PLUGINS_DIR "${SOFA_APPLICATIONS_DEV_DIR}/plugins" CACHE INTERNAL "Path to the Sofa applications-dev plugin directory")

## os-specific
if(WIN32)
	list(APPEND GLOBAL_COMPILER_DEFINES "UNICODE")
	set(SOFA_LIB_OS_DIR "${SOFA_LIB_DIR}/win32/Common" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
elseif(APPLE)
	set(SOFA_LIB_OS_DIR "${SOFA_LIB_DIR}/macx" CACHE INTERNAL "Path to the Sofa os-dependent lib directory")
endif()

# cached variables
set(GLOBAL_COMPILER_DEFINES ${GLOBAL_COMPILER_DEFINES} CACHE INTERNAL "Global Compiler Defines" FORCE)