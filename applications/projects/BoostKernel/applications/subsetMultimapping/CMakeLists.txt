cmake_minimum_required(VERSION 2.8)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    subsetMultimapping.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponentMain ObjectCreator SofaGuiMain SofaSimulationBGL)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

