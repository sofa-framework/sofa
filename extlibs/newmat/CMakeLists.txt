cmake_minimum_required(VERSION 2.8)

project("newmat")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(headers

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

set(sources

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

add_library(${PROJECT_NAME} STATIC ${headers} ${sources})
#target_link_libraries(${PROJECT_NAME})

#set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_DEFINITIONS "${GLOBAL_DEFINES};use_namespace")
	
include(${SOFA_CMAKE_DIR}/post.cmake)

