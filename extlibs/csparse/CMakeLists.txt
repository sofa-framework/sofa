include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    csparse.h 
    ldl.h 
    UFconfig.h 
	)

set(SOURCE_FILES

    csparse.c 
    ldl.c 
	)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

# The code must be relocatable if we want to link a shared library against it
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"
        OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    AddCompilerFlags("-fPIC")
endif()

include(${SOFA_CMAKE_DIR}/postProject.cmake)

