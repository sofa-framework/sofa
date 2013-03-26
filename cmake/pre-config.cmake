cmake_minimum_required(VERSION 2.8)

# useful variables
set(COMPILER_DEFINES "")
set(COMPILER_FLAGS "")
set(LINKER_DEPENDENCIES "")
set(LINKER_FLAGS "")

# include dir
include_directories("${SOFA_INC_DIR}")
include_directories("${SOFA_FRAMEWORK_DIR}")
include_directories("${SOFA_MODULES_DIR}")
include_directories("${SOFA_APPLICATIONS_DIR}")

if(MISC_USE_DEV_PROJECTS)
	include_directories("${SOFA_APPLICATIONS_DEV_DIR}")
endif()

if(EXTERNAL_BOOST_PATH)
	include_directories("${EXTERNAL_BOOST_PATH}")
else()
	include_directories("${SOFA_EXTLIBS_DIR}/miniBoost")
endif()

if(EXTERNAL_HAVE_EIGEN2)
	include_directories("${SOFA_EXTLIBS_DIR}/eigen-3.1.1")
endif()

## Zlib
if(EXTERNAL_HAVE_ZLIB)
	if(WIN32)
		set(ZLIB_LIBRARIES "zlib")
	else()
		find_library(ZLIB_LIBRARIES "z")
	endif()
	set(ZLIB_LIBRARIES ${ZLIB_LIBRARIES} CACHE INTERNAL "ZLib Library")
endif()

# lib dir
link_directories("${SOFA_LIB_DIR}")
link_directories("${SOFA_LIB_OS_DIR}")

# packages and libraries

## opengl / glew / glut
if(WIN32)
	set(OPENGL_LIBRARIES "opengl32")
	set(GLEW_LIBRARIES "glew")
	set(GLUT_LIBRARIES "glut32")
	set(PNG_LIBRARIES "libpng")
else()
	find_library(OPENGL_LIBRARIES "opengl")
	find_package(GLEW REQUIRED)
	find_library(GLUT_LIBRARIES "glut")
	find_library(PNG_LIBRARIES "png")
endif()

set(OPENGL_LIBRARIES ${OPENGL_LIBRARIES} CACHE INTERNAL "OpenGL Library")
set(GELW_LIBRARIES ${GELW_LIBRARIES} CACHE INTERNAL "GLEW Library")
set(GLUT_LIBRARIES ${GLUT_LIBRARIES} CACHE INTERNAL "GLUT Library")
set(PNG_LIBRARIES ${PNG_LIBRARIES} CACHE INTERNAL "PNG Library")


# target location
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SOFA_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${SOFA_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${SOFA_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SOFA_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${SOFA_BIN_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${SOFA_BIN_DIR}")
