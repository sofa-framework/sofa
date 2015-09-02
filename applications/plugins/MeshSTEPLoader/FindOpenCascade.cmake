# Find the OpenCascade headers and libraries.
#
#  OPENCASCADE_FOUND        - true if OpenCascade found.
#  OPENCASCADE_LIBRARIES    - the OpenCascade libraries
#  OPENCASCADE_INCLUDE_DIR  - where to find BRep_Toolbox.hxx, etc.
#
# On Windows, we also define 
#  OPENCASCADE_DLLS         - the OpenCascade DLLS
#
# If set, SOFA_OPENCASCADE_ROOT will be searched as well as standard locations.

if(SOFA_OPENCASCADE_ROOT)
    set(OPENCASCADE_ROOT ${SOFA_OPENCASCADE_ROOT})
elseif(ENV{SOFA_OPENCASCADE_ROOT})
    set(OPENCASCADE_ROOT $ENV{SOFA_OPENCASCADE_ROOT})
endif()

macro(find_opencascade_library lib)
    find_library(OPENCASCADE_${lib}_LIBRARY ${lib}
        PATHS
        ${OPENCASCADE_ROOT}
        ${OPENCASCADE_ROOT}/ros/win32/vc9/lib)

    if(OPENCASCADE_${lib}_LIBRARY)
        mark_as_advanced(OPENCASCADE_${lib}_LIBRARY)
        list(APPEND OPENCASCADE_LIBRARIES ${OPENCASCADE_${lib}_LIBRARY})
    else()
        set(OPENCASCADE_FOUND 0)
        message("Could not find OpenCascade library: ${lib}")
    endif()
endmacro()

find_path(OPENCASCADE_INCLUDE_DIR BRep_Tool.hxx
    PATHS
    ${OPENCASCADE_ROOT}
    ${OPENCASCADE_ROOT}/ros/inc
    PATH_SUFFIXES oce opencascade)

if(OPENCASCADE_INCLUDE_DIR)
    set(OPENCASCADE_FOUND 1)
endif()

set(OPENCASCADE_LIBRARIES "")

find_opencascade_library(TKBRep)
find_opencascade_library(TKernel)
find_opencascade_library(TKG3d)
find_opencascade_library(TKGeomBase)
find_opencascade_library(TKMath)
find_opencascade_library(TKMesh)
find_opencascade_library(TKPrim)
find_opencascade_library(TKSTEP)
find_opencascade_library(TKSTEPBase)
find_opencascade_library(TKShHealing)
find_opencascade_library(TKTopAlgo)
find_opencascade_library(TKXSBase)

mark_as_advanced(OPENCASCADE_LIBRARIES)
mark_as_advanced(OPENCASCADE_INCLUDE_DIR)

if(WIN32)
    file(GLOB core_libs "${OPENCASCADE_ROOT}/ros/win32/vc9/bin/*.dll")
    file(GLOB third_party_libs "${OPENCASCADE_ROOT}/3rdparty/tbb30_018oss/bin/ia32/vc9/*.dll")
    set(OPENCASCADE_DLLS ${core_libs} ${third_party_libs})
endif()

# Report the results.
if(NOT OPENCASCADE_FOUND)
    set(msg "Unable to find OpenCascade")
    if(OpenCascade_FIND_REQUIRED)
        message(FATAL_ERROR "${msg}")
    else()
        if(NOT OpenCascade_FIND_QUIETLY)
            message("${msg}")
        endif()
    endif()
else()
    if(NOT OpenCascade_FIND_QUIETLY)
        message(STATUS "Found OpenCascade: ${OPENCASCADE_TKBRep_LIBRARY}")
    endif()
endif()
