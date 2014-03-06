# This script locates the Sixense SDK
# ------------------------------------
#
# usage:
# find_package(Sixense ...)
#
# searches in SIXENSE_ROOT and usual locations
#
# Sets SIXENSE_INCLUDE_DIR, SIXENSE_LIBRARY_STATIC and SIXENSE_LIBRARY_DYNAMIC

set(SIXENSE_POSSIBLE_PATHS
	${SIXENSE_ROOT}
	$ENV{SIXENSE_ROOT}
  $ENV{SIXENSE_SDK_PATH}
	"$ENV{ProgramFiles}/Steam/steamapps/common/sixense sdk/SixenseSDK"
	"$ENV{ProgramFiles(x86)}/Steam/steamapps/common/sixense sdk/SixenseSDK"
	"$ENV{ProgramFiles}/SixenseSDK"
	"$ENV{ProgramFiles(x86)}/SixenseSDK"
	"C:/SixenseSDK"
	~/Library/Frameworks
	/Library/Frameworks
	/usr/local/
	/usr/
	/sw # Fink
	/opt/local/ # DarwinPorts
	/opt/csw/ # Blastwave
	/opt/
    ${CMAKE_CURRENT_SOURCE_DIR}
    $ENV{HOME} # user installation
	)


find_path(SIXENSE_INCLUDE_DIR sixense.h
	PATH_SUFFIXES
		"include"
	PATHS
		${SIXENSE_POSSIBLE_PATHS}
	)

find_library(SIXENSE_LIBRARY_STATIC sixense_s
	PATH_SUFFIXES
		"lib"
		"lib/win32/release_static"
	PATHS
		${SIXENSE_POSSIBLE_PATHS}
	)

find_library(SIXENSE_LIBRARY_DYNAMIC NAMES sixense_x64 sixense
	PATH_SUFFIXES
		"lib"
		"lib/win32/release_dll"
		"lib/linux/release"
	PATHS
		${SIXENSE_POSSIBLE_PATHS}
	)

find_library(SIXENSE_UTILS_LIBRARY NAMES sixense_utils_x64 sixense_utils
	PATH_SUFFIXES
		"lib"
		"lib/win32/release_dll"
		"lib/linux/release"
	PATHS
		${SIXENSE_POSSIBLE_PATHS}
	)
if(NOT SIXENSE_LIBRARY_DYNAMIC)
  set(SIXENSE_LIBRARY ${SIXENSE_LIBRARY_STATIC})
else(NOT SIXENSE_LIBRARY_DYNAMIC)
  set(SIXENSE_LIBRARY ${SIXENSE_LIBRARY_DYNAMIC})
endif(NOT SIXENSE_LIBRARY_DYNAMIC)

if ((NOT SIXENSE_INCLUDE_DIR) OR ((NOT SIXENSE_LIBRARY_STATIC) AND (NOT SIXENSE_LIBRARY_DYNAMIC)))
    if(Sixense_FIND_REQUIRED) #prefix is filename, case matters
        message(FATAL_ERROR "Could not find Sixense SDK!")
    elseif(NOT Sixense_FIND_QUIETLY)
        message("Could not find Sixense SDK!")
    endif(Sixense_FIND_REQUIRED)
endif ((NOT SIXENSE_INCLUDE_DIR) OR ((NOT SIXENSE_LIBRARY_STATIC) AND (NOT SIXENSE_LIBRARY_DYNAMIC)))

if (NOT Sixense_FIND_QUIETLY)
	message("Found Sixense SDK: Headers ${SIXENSE_INCLUDE_DIR} Dynamic lib ${SIXENSE_LIBRARY_DYNAMIC} Static lib ${SIXENSE_LIBRARY_STATIC} Utils lib ${SIXENSE_UTILS_LIBRARY}")
endif (NOT Sixense_FIND_QUIETLY)
