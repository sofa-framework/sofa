include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES
    config.h
)

set(SOURCE_FILES
    initEmptyCmakePlugin.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)