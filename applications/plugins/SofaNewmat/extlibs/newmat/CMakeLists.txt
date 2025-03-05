cmake_minimum_required(VERSION 3.22)
project(newmat VERSION 0.1)

set(HEADER_FILES
    newmat/boolean.h
    newmat/controlw.h
    newmat/include.h
    newmat/myexcept.h
    newmat/newmat.h
    newmat/newmatap.h
    newmat/newmatrc.h
    newmat/newmatrm.h
    newmat/precisio.h
)

set(SOURCE_FILES
    newmat/bandmat.cpp
    newmat/cholesky.cpp
    newmat/evalue.cpp
    newmat/fft.cpp
    newmat/hholder.cpp
    newmat/jacobi.cpp
    newmat/myexcept.cpp
    newmat/newfft.cpp
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
    newmat/sort.cpp
    newmat/submat.cpp
    newmat/svd.cpp
)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})
target_compile_definitions(${PROJECT_NAME} PUBLIC "use_namespace")
if(NOT DEFINED SOFA_WITH_DOUBLE)
    target_compile_definitions(${PROJECT_NAME} PUBLIC "USING_FLOAT")
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC "USING_DOUBLE")
endif()

if(WIN32)
    # remove warnings about deprecation (CRT,etc)
    target_compile_options(${PROJECT_NAME} PRIVATE "/wd4996")
endif()

# The code must be relocatable if we want to link a shared library against it
if("x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xGNU" OR "x${CMAKE_CXX_COMPILER_ID}" STREQUAL "xClang")
    target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif()


include(SofaMacros)
sofa_create_package_with_targets(
    PACKAGE_NAME Newmat
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "."
    INCLUDE_INSTALL_DIR "extlibs/Newmat"
    )
