cmake_minimum_required(VERSION 2.8)

cmake_policy(SET CMP0015 OLD)

# useful variables
set(COMPILER_DEFINES "")
set(COMPILER_FLAGS "")
set(LINKER_DEPENDENCIES "")
set(LINKER_FLAGS "")

## internal
set(ADDITIONAL_COMPILER_DEFINES "")
set(ADDITIONAL_LINKER_DEPENDENCIES "")

# include dir
if(WIN32)
    include_directories("${SOFA_INC_DIR}")
endif()
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
include_directories("${SOFA_EXTLIBS_DIR}/newmat")

if(EXTERNAL_HAVE_FLOWVR)
	include_directories("${SOFA_EXTLIBS_DIR}/miniFlowVR/include")
endif()

## Zlib (EXTERNAL_HAVE_ZLIB)
if(WIN32)
	set(ZLIB_LIBRARIES "zlib")
else()
	find_library(ZLIB_LIBRARIES "z")
endif()
set(ZLIB_LIBRARIES ${ZLIB_LIBRARIES} CACHE INTERNAL "ZLib Library")
RegisterDependencies(${ZLIB_LIBRARIES} OPTION EXTERNAL_HAVE_ZLIB)

# lib dir
link_directories("${SOFA_LIB_DIR}")
link_directories("${SOFA_LIB_OS_DIR}")

# packages and libraries

## opengl / glew / glut
find_package(OPENGL REQUIRED)
find_package(GLEW REQUIRED)
if(WIN32)
	#set(OPENGL_LIBRARIES "opengl32")
	#set(GLEW_LIBRARIES "glew")
	set(GLUT_LIBRARIES "glut32")
	set(PNG_LIBRARIES "libpng")
else()
	find_library(GLUT_LIBRARIES "glut")
        if(EXTERNAL_PNG_SPECIFIC_VERSION)
            set(PNG_LIBRARIES "${EXTERNAL_PNG_VERSION}")
        else()
            find_library(PNG_LIBRARIES "png")
        endif()
endif()

## GLU
if(UNIX)
    if(NOT APPLE)
        list(APPEND GLUT_LIBRARIES GLU X11)
    endif()
    list(APPEND GLUT_LIBRARIES dl)
endif()
list(REMOVE_DUPLICATES GLUT_LIBRARIES)

set(OPENGL_LIBRARIES ${OPENGL_LIBRARIES} CACHE INTERNAL "OpenGL Library")
set(GLEW_LIBRARIES ${GLEW_LIBRARIES} CACHE INTERNAL "GLEW Library")
set(GLUT_LIBRARIES ${GLUT_LIBRARIES} CACHE INTERNAL "GLUT Library")
set(PNG_LIBRARIES ${PNG_LIBRARIES} CACHE INTERNAL "PNG Library")

RegisterDependencies(${OPENGL_LIBRARIES})
RegisterDependencies(${GLEW_LIBRARIES} OPTION EXTERNAL_HAVE_GLEW)
RegisterDependencies(${GLUT_LIBRARIES})
RegisterDependencies(${PNG_LIBRARIES} OPTION EXTERNAL_HAVE_PNG)

## qt
set(ENV{QTDIR} "${EXTERNAL_QT_PATH}")
if(EXTERNAL_USE_QT4)
	set(ENV{CONFIG} "qt;uic;uic3")

	find_package(Qt4 COMPONENTS qtopengl qt3support qtxml REQUIRED)
else()
	set(ENV{CONFIG} "qt")
	
	find_package(Qt3 COMPONENTS qtopengl REQUIRED)
endif()
set(QT_QMAKE_EXECUTABLE ${QT_QMAKE_EXECUTABLE} CACHE INTERNAL "QMake executable path")
#message("${QT_LIBRARIES}")
#RegisterDependencies(${QT_LIBRARIES})

# target location
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SOFA_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${SOFA_BIN_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${SOFA_BIN_DIR}")
if(UNIX)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SOFA_LIB_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${SOFA_LIB_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${SOFA_LIB_DIR}")
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${SOFA_BIN_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${SOFA_BIN_DIR}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${SOFA_BIN_DIR}")
endif()

# out of source build support
set(CMAKE_INCLUDE_CURRENT_DIR 1)
