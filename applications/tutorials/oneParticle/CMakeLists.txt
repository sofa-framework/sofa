cmake_minimum_required(VERSION 2.8)

project("oneParticle")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(SOURCE_FILES

    oneParticle.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaSimulationTree SofaComponentMain SofaGuiMain)

include(${SOFA_CMAKE_DIR}/post.cmake)

