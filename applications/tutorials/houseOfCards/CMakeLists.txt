cmake_minimum_required(VERSION 2.8)

project("houseOfCards")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(SOURCE_FILES

    houseOfCards.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaSimulationTree SofaComponentMain SofaGuiMain)

include(${SOFA_CMAKE_DIR}/post.cmake)

