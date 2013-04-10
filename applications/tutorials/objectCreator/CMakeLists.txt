cmake_minimum_required(VERSION 2.8)

project("SofaObjectCreator")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    ObjectCreator.cpp

    )

set(HEADER_FILES

    ObjectCreator.h

    )

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES} ${HEADER_FILES})

set(LINKER_DEPENDENCIES SofaSimulationTree SofaComponentMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

