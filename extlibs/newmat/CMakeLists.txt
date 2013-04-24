cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

include_directories("${SOFA_BIN_DIR}")

set(HEADER_FILES

    newmat/include.h
    newmat/newmat.h
    newmat/newmatrc.h
    newmat/boolean.h
    newmat/myexcept.h
    newmat/controlw.h 
    newmat/newmatap.h
    newmat/newmatrm.h
    newmat/precisio.h 
	)

set(SOURCE_FILES

    newmat/newmat1.cpp
    newmat/newmat2.cpp
    newmat/newmat3.cpp
    newmat/newmat4.cpp
    newmat/newmat5.cpp
    newmat/newmat6.cpp
    newmat/newmat7.cpp
    newmat/newmat8.cpp
    newmat/newmat9.cpp
    newmat/newmatex.cpp
    newmat/newmatrm.cpp
    newmat/bandmat.cpp
    newmat/submat.cpp
    newmat/myexcept.cpp
    newmat/cholesky.cpp
    newmat/evalue.cpp
    newmat/fft.cpp
    newmat/hholder.cpp
    newmat/jacobi.cpp
    newmat/newfft.cpp
    newmat/sort.cpp
    newmat/svd.cpp 
	)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "use_namespace")

# fix link errors when linked with shared libraries
if(${CMAKE_COMPILER_IS_GNUCC})
    set(COMPILER_FLAGS "-fPIC")
endif()
	
include(${SOFA_CMAKE_DIR}/postProject.cmake)


