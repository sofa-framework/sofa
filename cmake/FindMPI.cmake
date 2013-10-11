# Try to find the METIS libraries
# MPI_FOUND - system has METIS lib
# MPI_INCLUDE_DIR - the METIS include directory
# MPI_LIBRARIES_DIR - Directory where the METIS libraries are located
# MPI_LIBRARIES - the METIS libraries




if(WIN32)
        ## todo
endif(WIN32)

if(UNIX)
        find_path(MPI_INCLUDE_DIR NAMES mpi.h PATHS "/usr/include/mpi")
        find_library(MPI_LIBRARY NAMES mpi PATHS "/usr/lib/")
        find_library(MPI_CXX_LIBRARY NAMES mpi_cxx PATHS "/usr/lib/")


        if(EXISTS ${MPI_INCLUDE_DIR} AND EXISTS ${MPI_LIBRARY} AND EXISTS ${MPI_CXX_LIBRARY})
                message(STATUS "MPI found : ${MPI_INCLUDE_DIR} ${MPI_LIBRARIES}")
                set(MPI_FOUND TRUE)
        else()
				message(FATAL_ERROR "MPI Lib not found: MPI_LIBRARIES=${MPI_LIBRARIES} MPI_INCLUDE_DIR=${MPI_INCLUDE_DIR}")
#                 message(ERROR "MPI NOT FOUND ${MPI_INCLUDE_DIR} ${MPI_LIBRARIES}")
                set(MPI_FOUND FALSE)
        endif()

endif(UNIX)




mark_as_advanced(MPI_INCLUDE_DIR METIS_LIBRARIES)
