if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(ASSIMP_ARCHITECTURE "64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
	set(ASSIMP_ARCHITECTURE "32")
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
	

find_package(Assimp NO_MODULE QUIET
    PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}/cmake
    )
    
if(NOT ASSIMP_INCLUDE_DIR OR NOT Assimp_FOUND)
    if(WIN32)
        set(ASSIMP_ROOT_DIR CACHE PATH "ASSIMP root directory")

        if(MSVC12)
            set(ASSIMP_MSVC_VERSION "vc120")
        elseif(MSVC14)	
            set(ASSIMP_MSVC_VERSION "vc140")
            set(ASSIMP_MSVC_VERSION2 "vc141")
        else()
            set(ASSIMP_MSVC_VERSION "")
        endif()
        
        
        FIND_PATH(ASSIMP_INCLUDE_DIR NAMES assimp/postprocess.h
          PATHS "C://Program Files//Assimp//include//" 
            ${ASSIMP_ROOT_DIR}//include
            ${ASSIMP_ROOT_DIR}/include
          DOC "The directory where assimp headers reside"
        )
        
        message(${ASSIMP_MSVC_VERSION})
        find_path(ASSIMP_LIBRARY_DIR
            NAMES assimp-${ASSIMP_MSVC_VERSION}-mt.lib assimp-${ASSIMP_MSVC_VERSION2}-mt.lib
            HINTS ${ASSIMP_ROOT_DIR}/lib${ASSIMP_ARCHITECTURE}
            ${ASSIMP_ROOT_DIR}/lib
        )
            
            
        find_library(ASSIMP_LIBRARY_RELEASE
            NAMES assimp-${ASSIMP_MSVC_VERSION}-mt.lib assimp-${ASSIMP_MSVC_VERSION2}-mt.lib
            PATHS 
              ${ASSIMP_LIBRARY_DIR}
              ${ASSIMP_ROOT_DIR}/lib
              "C://Program Files//Assimp" 
              "C://dev//Assimp//3.3.1"
            DOC "The assimp release library"
        )
        
        
        find_library(ASSIMP_LIBRARY_DEBUG
            NAMES assimp-${ASSIMP_MSVC_VERSION}-mtd.lib assimp-${ASSIMP_MSVC_VERSION2}-mtd.lib
            PATHS 
              ${ASSIMP_LIBRARY_DIR}
              ${ASSIMP_ROOT_DIR}/lib
               "C://Program Files//Assimp" 
               "C://dev//Assimp//3.3.1"
            DOC "The assimp debug library"
        )
            
            
        if (ASSIMP_LIBRARY_RELEASE)
            message("la 01") 
            if (ASSIMP_LIBRARY_DEBUG)
                set(ASSIMP_LIBRARY 
                    optimized 	${ASSIMP_LIBRARY_RELEASE}
                    debug		${ASSIMP_LIBRARY_DEBUG}
                    )
            else (ASSIMP_LIBRARY_DEBUG)
                set(ASSIMP_LIBRARY ${ASSIMP_LIBRARY_RELEASE})
            endif()
            
            if(ASSIMP_INCLUDE_DIR)
            message("la 02")
                string(REPLACE "/include" "" ASSIMP_ROOT_DIR ${ASSIMP_INCLUDE_DIR})
                set(ASSIMP_BIN_DIR "${ASSIMP_ROOT_DIR}/bin/")
                set(ASSIMP_DLL ${ASSIMP_ROOT_DIR}/bin/assimp-${ASSIMP_MSVC_VERSION}-mt.dll)
                
                SET(Assimp_FOUND TRUE)
            endif()            
        else()
            message("la 03")
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

message("Assimp_FOUND: " ${Assimp_FOUND})
message("assimp_FIND_QUIETLY: " ${Assimp_FIND_QUIETLY})
message("assimp_FIND_REQUIRED:  ${${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED}")

message("CMAKE_FIND_PACKAGE_NAME: " ${CMAKE_FIND_PACKAGE_NAME})

if (${CMAKE_FIND_PACKAGE_NAME}_FOUND)
message("la 1")
  if (NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
    message(STATUS "Found asset importer library: ${ASSIMP_LIBRARY}")
  endif (NOT ${CMAKE_FIND_PACKAGE_NAME}_FIND_QUIETLY)
else (${CMAKE_FIND_PACKAGE_NAME}_FOUND)
message("la 2")
  if (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find asset importer library")
  endif (${CMAKE_FIND_PACKAGE_NAME}_FIND_REQUIRED)
endif (${CMAKE_FIND_PACKAGE_NAME}_FOUND)
