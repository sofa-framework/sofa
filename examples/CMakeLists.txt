cmake_minimum_required(VERSION 3.22)
project(SofaScenes)

file(GLOB_RECURSE EXAMPLES_FILES ../*.scn ../*.pscn ../*.pyscn)

add_library(${PROJECT_NAME} INTERFACE ${EXAMPLES_FILES})

# Detect if component is compiled outside SOFA
if ("${CMAKE_PROJECT_NAME}" STREQUAL "${PROJECT_NAME}")
    install(DIRECTORY . DESTINATION share/sofa/examples COMPONENT resources)
    install(DIRECTORY ../share/ DESTINATION share/sofa COMPONENT resources)
endif()
