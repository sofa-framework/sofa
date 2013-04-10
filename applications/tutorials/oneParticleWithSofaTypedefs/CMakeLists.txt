cmake_minimum_required(VERSION 2.8)

project("oneParticleWithSofaTypedefs")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    oneParticleWithSofaTypedefs.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaSimulationTree SofaComponentMain SofaGuiMain)

include(${SOFA_CMAKE_DIR}/postProject.cmake)

