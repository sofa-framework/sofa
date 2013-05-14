include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    ObjectCreator.cpp
    )

set(HEADER_FILES

    ObjectCreator.h
    )

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_OBJECTCREATOR")
set(LINKER_DEPENDENCIES SofaSimulationTree SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

