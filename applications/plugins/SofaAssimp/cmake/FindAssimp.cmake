if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(ASSIMP_ARCHITECTURE "64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(ASSIMP_ARCHITECTURE "32")
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
	
# First try to find install of assimp on the system
find_package(Assimp NO_MODULE QUIET
    PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake
    )
    
if(NOT ASSIMP_INCLUDE_DIR OR NOT Assimp_FOUND) #If not found, will try to manually find it using ASSIMP_ROOT_DIR
    message(STATUS "No install of Assimp library found using cmake find_package default location : /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake , please specify manually the {ASSIMP_ROOT_DIR}")
    if(WIN32)
        set(ASSIMP_ROOT_DIR CACHE PATH "ASSIMP root directory")
        
        FIND_PATH(ASSIMP_INCLUDE_DIR 
          NAMES assimp/postprocess.h
          PATHS "C://Program Files//Assimp//include//" 
            ${ASSIMP_ROOT_DIR}//include
            ${ASSIMP_ROOT_DIR}/include
          DOC "The directory where assimp headers reside"
        )
        
        
        if(MSVC12)
            set(ASSIMP_MSVC_VERSION "vc120")
        elseif(MSVC14)	
            
            #First look for vc140 build and if not found, will try vc141
            find_path(ASSIMP_LIBRARY_DIR
                NAMES assimp-vc140-mt.lib
                HINTS ${ASSIMP_ROOT_DIR}/lib${ASSIMP_ARCHITECTURE}
                ${ASSIMP_ROOT_DIR}/lib
                ${ASSIMP_ROOT_DIR}/lib/x64
            )
            
            if(ASSIMP_LIBRARY_DIR)
                set(ASSIMP_MSVC_VERSION "vc140")
            else()
                set(ASSIMP_MSVC_VERSION "vc141")
            endif()
        else()
            set(ASSIMP_MSVC_VERSION "")
        endif()

        if(NOT ASSIMP_LIBRARY_DIR)
            find_path(ASSIMP_LIBRARY_DIR
                NAMES assimp-${ASSIMP_MSVC_VERSION}-mt.lib
                HINTS ${ASSIMP_ROOT_DIR}/lib${ASSIMP_ARCHITECTURE}
                ${ASSIMP_ROOT_DIR}/lib
                ${ASSIMP_ROOT_DIR}/lib/x64
            )        
        endif()
            
        find_library(ASSIMP_LIBRARY_RELEASE
            NAMES assimp-${ASSIMP_MSVC_VERSION}-mt.lib
            PATHS 
              ${ASSIMP_LIBRARY_DIR}
              ${ASSIMP_ROOT_DIR}/lib
              "C://Program Files//Assimp" 
              "C://dev//Assimp//3.3.1"
            DOC "The assimp release library"
        )
        
        
        find_library(ASSIMP_LIBRARY_DEBUG
            NAMES assimp-${ASSIMP_MSVC_VERSION}-mtd.lib
            PATHS 
              ${ASSIMP_LIBRARY_DIR}
              ${ASSIMP_ROOT_DIR}/lib
               "C://Program Files//Assimp" 
               "C://dev//Assimp//3.3.1"
            DOC "The assimp debug library"
        )
         

        find_path(ASSIMP_BIN_DIR
            NAMES assimp-${ASSIMP_MSVC_VERSION}-mt.dll
            HINTS ${ASSIMP_ROOT_DIR}/bin${ASSIMP_ARCHITECTURE}
            ${ASSIMP_ROOT_DIR}/bin
            ${ASSIMP_ROOT_DIR}/bin/x64
        )
            
        if (ASSIMP_LIBRARY_RELEASE AND ASSIMP_BIN_DIR)
            if (ASSIMP_LIBRARY_DEBUG)
                set(ASSIMP_LIBRARY 
                    optimized 	${ASSIMP_LIBRARY_RELEASE}
                    debug		${ASSIMP_LIBRARY_DEBUG}
                    )
            else (ASSIMP_LIBRARY_DEBUG)
                set(ASSIMP_LIBRARY ${ASSIMP_LIBRARY_RELEASE})
            endif()

            set(ASSIMP_DLL ${ASSIMP_BIN_DIR}/assimp-${ASSIMP_MSVC_VERSION}-mt.dll)
            set(Assimp_FOUND TRUE)
        else()
            SET(Assimp_FOUND FALSE)
        endif()
        
        
            
    else(WIN32)
        find_path(ASSIMP_INCLUDE_DIR
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
        
        if (ASSIMP_INCLUDE_DIR AND ASSIMP_LIBRARY)
          SET(Assimp_FOUND TRUE)
        endif (ASSIMP_INCLUDE_DIR AND ASSIMP_LIBRARY)
        
    endif(WIN32)
endif(NOT ASSIMP_INCLUDE_DIR OR NOT Assimp_FOUND)

if (${CMAKE_FIND_PACKAGE_NAME}_FOUND)
  if (NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
    message(STATUS "Found Assimp library: ${ASSIMP_LIBRARY} and include directory: ${ASSIMP_INCLUDE_DIR}")
  endif (NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
else (${CMAKE_FIND_PACKAGE_NAME}_FOUND)
  if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find asset importer library")
  endif (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
endif (${CMAKE_FIND_PACKAGE_NAME}_FOUND)
