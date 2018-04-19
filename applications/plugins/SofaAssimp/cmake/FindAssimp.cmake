if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(ASSIMP_ARCHITECTURE "64")
elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(ASSIMP_ARCHITECTURE "32")
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
    
if(WIN32)
    set(ASSIMP_ROOT_DIR CACHE PATH "ASSIMP root directory")
 message("ASSIMP_ROOT_DIR: ${ASSIMP_ROOT_DIR}")
    FIND_PATH( ASSIMP_INCLUDE_DIR 
      NAMES assimp/postprocess.h assimp/scene.h assimp/version.h assimp/config.h assimp/cimport.h
      PATHS 
        "C://Program Files//Assimp//include//" 
        ${ASSIMP_ROOT_DIR}/include/
		DOC "The directory where assimp headers reside")
	FIND_LIBRARY( ASSIMP_LIBRARY_RELEASE
		NAMES assimp
		PATH_SUFFIXES lib/x64
        PATHS
            "C://Program Files//Assimp" 
            "C://dev//Assimp//3.3.1"
            ${ASSIMP_ROOT_DIR}
		DOC "The assimp library")
	FIND_LIBRARY( ASSIMP_LIBRARY_DEBUG
		NAMES assimp
		PATH_SUFFIXES lib/x64
        PATHS
            "C://Program Files//Assimp" 
            "C://dev//Assimp//3.3.1"
            ${ASSIMP_ROOT_DIR}
		DOC "The assimp library")
		
        
	IF(ASSIMP_LIBRARY_RELEASE)
		SET(ASSIMP_LIBRARY_DIR debug ${ASSIMP_LIBRARY_DEBUG} optimized ${ASSIMP_LIBRARY_RELEASE})
	ELSEIF(ASSIMP_LIBRARY_DEBUG)
		SET(ASSIMP_LIBRARY_DIR ${ASSIMP_LIBRARY_DEBUG})
	ENDIF()

    SET(ASSIMP_MSVC_VERSION "vc140")
    FUNCTION(ASSIMP_COPY_BINARIES TargetDirectory)
        ADD_CUSTOM_TARGET(AssimpCopyBinaries
            COMMAND ${CMAKE_COMMAND} -E copy ${ASSIMP_ROOT_DIR}/bin/${ASSIMP_ARCHITECTURE}/assimp-${ASSIMP_MSVC_VERSION}-mtd.dll     ${TargetDirectory}/Debug/assimp-${ASSIMP_MSVC_VERSION}-mtd.dll
            COMMAND ${CMAKE_COMMAND} -E copy ${ASSIMP_ROOT_DIR}/bin/${ASSIMP_ARCHITECTURE}/assimp-${ASSIMP_MSVC_VERSION}-mt.dll         ${TargetDirectory}/Release/assimp-${ASSIMP_MSVC_VERSION}-mt.dll
        COMMENT "Copying Assimp binaries to '${TargetDirectory}'"
        VERBATIM)
    ENDFUNCTION(ASSIMP_COPY_BINARIES)

    
else(WIN32)
    find_path(ASSIMP_INCLUDE_DIR
      NAMES assimp/postprocess.h assimp/scene.h assimp/version.h assimp/config.h assimp/cimport.h
      PATHS 
      /usr/include
      /usr/local/include
      /sw/include
      /opt/local/include
      DOC "The directory where assimp headers reside")


    find_library(ASSIMP_LIBRARY_DIR
      NAMES assimp
      PATHS 
      /usr/local/lib/
      /usr/lib64
      /usr/lib
      /usr/local/lib64
      /usr/local/lib
      /sw/lib
      /opt/local/lib
	  DOC "The assimp library")
    
endif(WIN32)

if (ASSIMP_INCLUDE_DIR AND ASSIMP_LIBRARY_DIR)
  SET(assimp_FOUND TRUE)
  message("ASSIMP_INCLUDE_DIR: ${ASSIMP_INCLUDE_DIR}")
  message("ASSIMP_LIBRARY_DIR: ${ASSIMP_LIBRARY_DIR}")
ENDIF (ASSIMP_INCLUDE_DIR AND ASSIMP_LIBRARY_DIR)

if (assimp_FOUND)
  if (NOT assimp_FIND_QUIETLY)
    message(STATUS "Found Assimp importer library: ${ASSIMP_LIBRARY_DIR}")
  endif (NOT assimp_FIND_QUIETLY)
else (assimp_FOUND)
    message(FATAL_ERROR "Could not find Assimp library")
endif (assimp_FOUND)