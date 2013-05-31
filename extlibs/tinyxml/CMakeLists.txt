include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

    tinystr.h
    tinyxml.h
	)

set(SOURCE_FILES

    tinystr.cpp
    tinyxml.cpp
    tinyxmlerror.cpp
    tinyxmlparser.cpp
	)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_TINYXML" "TIXML_USE_STL")

include(${SOFA_CMAKE_DIR}/postProject.cmake)

