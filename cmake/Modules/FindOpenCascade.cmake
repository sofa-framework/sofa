# - Find OpenCascade
# Find the OpenCascade headers and libraries.
#
#  OPENCASCADE_INCLUDE_DIR -  where to find BRep_Toolbox.hxx, etc.
#  OPENCASCADE_LIBRARIES_DIR - where to find all the libraries
#  OPENCASCADE_FOUND        - True if OpenCascade found.

if(WIN32)
    find_path(OPENCASCADE_INCLUDE_DIR BRep_Tool.hxx
        PATHS ${SOFA_EXTLIBS_DIR}/OpenCASCADE*/ros/inc )
    find_path(OPENCASCADE_LIBRARIES_DIR TKBRep.lib
        PATHS ${SOFA_EXTLIBS_DIR}/OpenCASCADE*/ros/win32/vc9/lib )
else()
    # Search standard locations
    find_path(OPENCASCADE_INCLUDE_DIR BRep_Tool.hxx PATH_SUFFIXES oce opencascade)
    find_library(OPENCASCADE_LIBRARIES TKBRep)
    mark_as_advanced(OPENCASCADE_LIBRARIES)
    get_filename_component(OPENCASCADE_LIBRARIES_DIR ${OPENCASCADE_LIBRARIES} PATH)
endif()
mark_as_advanced(OPENCASCADE_INCLUDE_DIR)
mark_as_advanced(OPENCASCADE_LIBRARIES_DIR)

if(OPENCASCADE_INCLUDE_DIR AND OPENCASCADE_LIBRARIES_DIR)
    set(OPENCASCADE_FOUND 1)
endif()

# Report the results.
if(NOT OPENCASCADE_FOUND)
    set(msg "Unable to find OpenCascade.")
    if(OpenCascade_FIND_REQUIRED)
        message(FATAL_ERROR "${msg}")
    else()
        message("${msg}")
    endif()
endif()
