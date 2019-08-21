# - Find OpenHaptics
# Find the native OPENHAPTICS headers and libraries.
#
#  OPENHAPTICS_INCLUDE_DIR -  where to find OpenHaptics.h, etc.
#  OPENHAPTICS_UTIL_INCLUDE_DIR - where to find HDU/hdu.h, etc.
#  OPENHAPTICS_LIBRARIES    - List of libraries when using OpenHaptics.
#  OPENHAPTICS_FOUND        - True if OpenHaptics found.

SET( program_files_path "" )
IF( CMAKE_CL_64 )
    SET( LIB "x64" )
    SET( program_files_path "$ENV{ProgramW6432}" )
ELSE( CMAKE_CL_64 )
    SET( LIB "win32" )
    SET( program_files_path "$ENV{ProgramFiles}" )
ENDIF( CMAKE_CL_64 )

# Look for the header file.
FIND_PATH(OPENHAPTICS_INCLUDE_DIR NAMES HL/hl.h HD/hd.h HDU/hdu.h
    PATHS $ENV{3DTOUCH_BASE}/include
    $ENV{OH_SDK_BASE}/include
    "${program_files_path}/SensAble/3DTouch/include"
    DOC "Path in which the files HL/hl.h and HD/hd.h are located." )
MARK_AS_ADVANCED(OPENHAPTICS_INCLUDE_DIR)

FIND_PATH(OPENHAPTICS_UTIL_INCLUDE_DIR NAMES HDU/hdu.h
    HINTS $ENV{3DTOUCH_BASE}/utilities/include
    $ENV{OH_SDK_BASE}/utilities/include
    "${program_files_path}/SensAble/3DTouch/utilities/include"
    DOC "Path in which the files HDU/hdu.h are located." )
MARK_AS_ADVANCED(OPENHAPTICS_UTIL_INCLUDE_DIR)


SET( OPENHAPTICS_LIBRARY_DIRECTORIES $ENV{3DTOUCH_BASE}/lib        # OpenHaptics 2.0
    $ENV{3DTOUCH_BASE}/lib/${LIB}  # OpenHaptics 3.0
    $ENV{OH_SDK_BASE}/lib        # OpenHaptics 2.0
    $ENV{OH_SDK_BASE}/lib/${LIB}  # OpenHaptics 3.0
    $ENV{OH_SDK_BASE}/lib/${LIB}/Release
    $ENV{OH_SDK_BASE}/lib/${LIB}/ReleaseAcademicEdition
    "${program_files_path}/SensAble/3DTouch/lib"        # OpenHaptics 2.0
    "${program_files_path}/SensAble/3DTouch/lib/${LIB}" # OpenHaptics 3.0
    "/usr/lib64" )

# TODO: Add conditional checking for x64 system
# Look for the library.
FIND_LIBRARY(OPENHAPTICS_HL_LIBRARY NAMES HL
    PATHS ${OPENHAPTICS_LIBRARY_DIRECTORIES}
    DOC "Path to hl library." )

MARK_AS_ADVANCED(OPENHAPTICS_HL_LIBRARY)

FIND_LIBRARY(OPENHAPTICS_HD_LIBRARY NAMES HD
    PATHS ${OPENHAPTICS_LIBRARY_DIRECTORIES}
    DOC "Path to hd library." )
MARK_AS_ADVANCED(OPENHAPTICS_HD_LIBRARY)

FIND_LIBRARY(OPENHAPTICS_HDU_LIBRARY NAMES HDU
    PATHS  $ENV{3DTOUCH_BASE}/utilities/lib        # OpenHaptics 2.0
    $ENV{3DTOUCH_BASE}/utilities/lib/${LIB}/Release  # OpenHaptics 3.0
    $ENV{OH_SDK_BASE}/utilities/lib        # OpenHaptics 2.0
    $ENV{OH_SDK_BASE}/utilities/lib/${LIB}/Release  # OpenHaptics 3.0
    "${program_files_path}/SensAble/3DTouch/utilities/lib"        # OpenHaptics 2.0
    "${program_files_path}/SensAble/3DTouch/utilities/lib/${LIB}/Release"  # OpenHaptics 3.0
    "/usr/lib64"
    DOC "Path to hdu library." )
MARK_AS_ADVANCED(OPENHAPTICS_HDU_LIBRARY)

# Copy the results to the output variables.
IF(OPENHAPTICS_INCLUDE_DIR AND OPENHAPTICS_HD_LIBRARY AND OPENHAPTICS_HL_LIBRARY AND OPENHAPTICS_HDU_LIBRARY)
    SET(OPENHAPTICS_FOUND 1)
    SET(OPENHAPTICS_LIBRARIES ${OPENHAPTICS_HD_LIBRARY} ${OPENHAPTICS_HL_LIBRARY} ${OPENHAPTICS_HDU_LIBRARY})
    SET(OPENHAPTICS_INCLUDE_DIR ${OPENHAPTICS_INCLUDE_DIR})
ELSE(OPENHAPTICS_INCLUDE_DIR AND OPENHAPTICS_HD_LIBRARY AND OPENHAPTICS_HL_LIBRARY AND OPENHAPTICS_HDU_LIBRARY)
    SET(OPENHAPTICS_FOUND 0)
    SET(OPENHAPTICS_LIBRARIES)
    SET(OPENHAPTICS_INCLUDE_DIR)
ENDIF(OPENHAPTICS_INCLUDE_DIR  AND OPENHAPTICS_HD_LIBRARY AND OPENHAPTICS_HL_LIBRARY AND OPENHAPTICS_HDU_LIBRARY)

# Report the results.
IF(NOT OPENHAPTICS_FOUND)
    SET(OPENHAPTICS_DIR_MESSAGE
        "OPENHAPTICS [hapi] was not found. Make sure to set OPENHAPTICS_HL_LIBRARY, OPENHAPTICS_HD_LIBRARY, OPENHAPTICS_HDU_LIBRARY and OPENHAPTICS_INCLUDE_DIR. If you do not have it you will not be able to use haptics devices from SensAble Technologies such as the Phantom.")
    IF(OpenHaptics_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "${OPENHAPTICS_DIR_MESSAGE}")
    ELSEIF(NOT OpenHaptics_FIND_QUIETLY)
        MESSAGE(STATUS "${OPENHAPTICS_DIR_MESSAGE}")
    ENDIF(OpenHaptics_FIND_REQUIRED)
ENDIF(NOT OPENHAPTICS_FOUND)
