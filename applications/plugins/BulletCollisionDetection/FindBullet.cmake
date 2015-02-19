# Locate Bullet.
#
# This script defines:
# BULLET_FOUND, set to 1 if found
# BULLET_LIBRARIES
# BULLET_INCLUDE_DIR
# BULLET_*_LIBRARY, one for each library (for example, "BULLET_BulletCollision_LIBRARY").
# BULLET_*_LIBRARY_debug, one for each library.
#
# This script will look in standard locations for installed Bullet. However, if
# you install Bullet into a non-standard location, you can use the BULLET_ROOT
# variable (in environment or CMake) to specify the location.

if(NOT BULLET_ROOT)
    if(NOT "$ENV{BULLET_ROOT}" STREQUAL "")
        set(BULLET_ROOT "$ENV{BULLET_ROOT}" CACHE PATH "Bullet install prefix")
    endif()
endif()

unset(BULLET_INCLUDE_DIR CACHE)
mark_as_advanced(BULLET_INCLUDE_DIR)
find_path(BULLET_INCLUDE_DIR btBulletCollisionCommon.h
    PATHS
    ${BULLET_ROOT}
    NO_DEFAULT_PATH
    PATH_SUFFIXES
    /include
    /include/bullet)

macro(FIND_BULLET_LIBRARY_DIRNAME LIBNAME DIRNAME)
    unset(BULLET_${LIBNAME}_LIBRARY CACHE)
    unset(BULLET_${LIBNAME}_LIBRARY_debug CACHE)
    mark_as_advanced(BULLET_${LIBNAME}_LIBRARY)
    mark_as_advanced(BULLET_${LIBNAME}_LIBRARY_debug)
    find_library(BULLET_${LIBNAME}_LIBRARY
        NAMES
        ${LIBNAME}
        PATHS
        ${BULLET_ROOT}
        NO_DEFAULT_PATH
        PATH_SUFFIXES
        ./src/${DIRNAME}
        ./Extras/${DIRNAME}
        ./Demos/${DIRNAME}
        ./src/${DIRNAME}/Release
        ./Extras/${DIRNAME}/Release
        ./Demos/${DIRNAME}/Release
        ./libs/${DIRNAME}
        ./libs
        ./lib
        ./lib/Release) # v2.76, new location for build tree libs on Windows
    find_library(BULLET_${LIBNAME}_LIBRARY_debug
        NAMES
        ${LIBNAME}
        PATHS
        ${BULLET_ROOT}
        NO_DEFAULT_PATH
        PATH_SUFFIXES
        ./src/${DIRNAME}
        ./Extras/${DIRNAME}
        ./Demos/${DIRNAME}
        ./src/${DIRNAME}/Debug
        ./Extras/${DIRNAME}/Debug
        ./Demos/${DIRNAME}/Debug
        ./libs/${DIRNAME}
        ./libs
        ./lib
        ./lib/Debug) # v2.76, new location for build tree libs on Windows
    if(BULLET_${LIBNAME}_LIBRARY)
        set(BULLET_LIBRARIES ${BULLET_LIBRARIES}
            "optimized" ${BULLET_${LIBNAME}_LIBRARY})
    endif(BULLET_${LIBNAME}_LIBRARY)
    if(BULLET_${LIBNAME}_LIBRARY_debug)
        set(BULLET_LIBRARIES ${BULLET_LIBRARIES}
            "debug" ${BULLET_${LIBNAME}_LIBRARY_debug})
    endif(BULLET_${LIBNAME}_LIBRARY_debug)
endmacro(FIND_BULLET_LIBRARY_DIRNAME LIBNAME)

macro(FIND_BULLET_LIBRARY LIBNAME)
    find_bullet_library_dirname(${LIBNAME} ${LIBNAME})
endmacro(FIND_BULLET_LIBRARY LIBNAME)


find_bullet_library(BulletDynamics)
find_bullet_library(BulletSoftBody)
find_bullet_library(BulletCollision)
find_bullet_library(BulletMultiThreaded)
find_bullet_library(LinearMath)
find_bullet_library(HACD)
find_bullet_library_dirname(OpenGLSupport OpenGL)

# Hide BULLET_LIBRARY in the GUI, since most users can just ignore it
mark_as_advanced(BULLET_LIBRARIES)
mark_as_advanced(BULLET_LIBRARIES_debug)

if(BULLET_INCLUDE_DIR AND BULLET_LIBRARIES)
    set(BULLET_FOUND 1)
else()
    set(BULLET_FOUND 0)
    message(FATAL_ERROR "Could not find Bullet, please install bullet and set BULLET_ROOT to the install prefix.")
endif()
