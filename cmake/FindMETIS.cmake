# Try to find the METIS libraries
# METIS_FOUND - system has METIS lib
# METIS_INCLUDE_DIR - the METIS include directory
# METIS_LIBRARIES - the METIS libraries




if(WIN32)
        ## todo
        #set(GLEW_LIBRARIES "glew32")
        #set(GLEW_FOUND CACHE INTERNAL TRUE)
endif(WIN32)

if(UNIX)
        find_path(METIS_INCLUDE_DIR NAMES metis.h)
        if(NOT EXISTS "${METIS_INCLUDE_DIR}/metis.h")
            # on some systems metis.h is located in /usr/include/metis...
            find_path(METIS_INCLUDE_DIR NAMES metis.h PATHS /usr/include/metis /opt/local/include/metis)
        endif()
        find_library(METIS_LIBRARIES NAMES metis)

        #message(STATUS "METIS Lib found: METIS_LIBRARIES=${METIS_LIBRARIES} METIS_INCLUDE_DIR=${METIS_INCLUDE_DIR}")

        if(EXISTS "${METIS_INCLUDE_DIR}/metis.h" AND EXISTS ${METIS_LIBRARIES})
                #if (NOT DEFINED METIS_FOUND)
                        message(STATUS "METIS found: ${METIS_LIBRARIES} ${METIS_INCLUDE_DIR}")
                #endif(NOT DEFINED METIS_FOUND)
                set(METIS_FOUND CACHE INTERNAL TRUE)
        else()
                message(ERROR " - METIS NOT FOUND! ${METIS_LIBRARIES} ${METIS_INCLUDE_DIR}")
                set(METIS_FOUND FALSE)
        endif()

endif(UNIX)




mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARIES)
