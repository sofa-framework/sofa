if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(ASSIMP_ARCHITECTURE "64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(ASSIMP_ARCHITECTURE "32")
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
	
# First try to find assimp in CONFIG mode on the system
find_package(assimp NO_MODULE QUIET)


# If not found, try to manually find it
if(NOT ASSIMP_INCLUDE_DIRS OR NOT assimp_FOUND)
    if(CMAKE_SYSTEM_NAME STREQUAL Windows)
        # Use ASSIMP_ROOT_DIR as user input for Assimp location
        set(ASSIMP_ROOT_DIR CACHE PATH "Assimp root directory")

        find_path(ASSIMP_INCLUDE_DIRS
            NAMES assimp/postprocess.h
            PATHS ${ASSIMP_ROOT_DIR}/include
            DOC "The directory where assimp headers reside"
            )

        if(MSVC12)
            set(ASSIMP_MSVC_VERSION "vc120")
        elseif(MSVC14)
            # First look for vc140 build and if not found try vc141
            find_path(ASSIMP_LIBRARY_DIR
                NAMES assimp-vc140-mt.lib
                HINTS
                    ${ASSIMP_ROOT_DIR}/lib${ASSIMP_ARCHITECTURE}
                    ${ASSIMP_ROOT_DIR}/lib
                    ${ASSIMP_ROOT_DIR}/lib/x64
                )

            if(ASSIMP_LIBRARY_DIR)
                set(ASSIMP_MSVC_VERSION "vc140")
            else()
                set(ASSIMP_MSVC_VERSION "vc141")
            endif()
		elseif (MSVC17)
			set(ASSIMP_MSVC_VERSION "vc143")
        else()
            set(ASSIMP_MSVC_VERSION "")
        endif()

        if(NOT ASSIMP_LIBRARY_DIR)
            find_path(ASSIMP_LIBRARY_DIR
                NAMES assimp-${ASSIMP_MSVC_VERSION}-mt.lib
                HINTS
                    ${ASSIMP_ROOT_DIR}/lib${ASSIMP_ARCHITECTURE}
                    ${ASSIMP_ROOT_DIR}/lib
                    ${ASSIMP_ROOT_DIR}/lib/x64
                )
        endif()

        find_library(ASSIMP_LIBRARY_DEBUG
            NAMES assimp-${ASSIMP_MSVC_VERSION}-mtd.lib
            PATHS
                ${ASSIMP_LIBRARY_DIR}
                ${ASSIMP_ROOT_DIR}/lib
            DOC "The assimp debug library"
            )

        find_library(ASSIMP_LIBRARY_RELEASE
            NAMES assimp-${ASSIMP_MSVC_VERSION}-mt.lib
            PATHS
                ${ASSIMP_LIBRARY_DIR}
                ${ASSIMP_ROOT_DIR}/lib
            DOC "The assimp release library"
            )

        find_path(ASSIMP_BIN_DIR
            NAMES assimp-${ASSIMP_MSVC_VERSION}-mt.dll
            HINTS
                ${ASSIMP_ROOT_DIR}/bin${ASSIMP_ARCHITECTURE}
                ${ASSIMP_ROOT_DIR}/bin
                ${ASSIMP_ROOT_DIR}/bin/x64
            )

        if(ASSIMP_LIBRARY_RELEASE AND ASSIMP_BIN_DIR)
            if(ASSIMP_LIBRARY_DEBUG)
                set(ASSIMP_LIBRARY
                    optimized 	${ASSIMP_LIBRARY_RELEASE}
                    debug		${ASSIMP_LIBRARY_DEBUG}
                    )
            else()
                set(ASSIMP_LIBRARY ${ASSIMP_LIBRARY_RELEASE})
            endif()

            set(ASSIMP_DLL ${ASSIMP_BIN_DIR}/assimp-${ASSIMP_MSVC_VERSION}-mt.dll)
            set(assimp_FOUND TRUE)
        else()
            set(assimp_FOUND FALSE)
        endif()
    else()
        find_path(ASSIMP_INCLUDE_DIRS
            NAMES assimp/postprocess.h assimp/scene.h assimp/version.h assimp/config.h assimp/cimport.h
            PATHS
                /usr/include
                /usr/local/include
                /sw/include
                /opt/local/include
            DOC "The directory where assimp headers reside"
            )

        find_library(ASSIMP_LIBRARY
            NAMES assimp
            PATHS
                /usr/local/lib/
                /usr/lib64
                /usr/lib
                /usr/local/lib64
                /usr/local/lib
                /sw/lib
                /opt/local/lib
            DOC "The assimp library"
            )

        if(ASSIMP_INCLUDE_DIRS AND ASSIMP_LIBRARY)
            set(assimp_FOUND TRUE)
        endif()
    endif()
endif()

if(assimp_FOUND)
    if(NOT assimp_FIND_QUIETLY)
      message(STATUS "Assimp found. Library is ${ASSIMP_LIBRARIES} and include dir is ${ASSIMP_INCLUDE_DIRS}")
    endif()
else()
    if(${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
        set(error_message "Assimp not found.")
        if(CMAKE_SYSTEM_NAME STREQUAL Windows AND NOT ASSIMP_ROOT_DIR)
            set(error_message "${error_message} Please set ASSIMP_ROOT_DIR to locate Assimp.")
        endif()
        message(FATAL_ERROR "${error_message}")
    endif()
endif()
