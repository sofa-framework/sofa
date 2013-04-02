cmake_minimum_required(VERSION 2.8)

project("newmat")

include(${SOFA_CMAKE_DIR}/pre.cmake)

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

# fix link errors when linked with shared libraries
if("${CMAKE_SYSTEM_PROCESSOR}" STREQUAL "x86_64_32")
    set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS -fPIC)
endif()

set(COMPILER_DEFINES "use_namespace")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

