cmake_minimum_required(VERSION 2.8)

project("chainHybrid")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(SOURCE_FILES

    chainHybrid.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaSimulationTree SofaComponentMain SofaGuiMain SofaObjectCreator)

include(${SOFA_CMAKE_DIR}/post.cmake)

