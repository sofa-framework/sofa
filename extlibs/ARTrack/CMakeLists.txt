include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    dtracklib.h
	)

set(SOURCE_FILES

    mainTracker.cpp
	)

add_library(${PROJECT_NAME} STATIC ${HEADER_FILES} ${SOURCE_FILES})

# fix link errors when linked with shared libraries
if(${CMAKE_COMPILER_IS_GNUCC})
    AddCompilerFlags("-fPIC")
endif()

AddCompilerDefinitions("use_namespace")

include(${SOFA_CMAKE_DIR}/postProject.cmake)

