cmake_minimum_required(VERSION 2.8)

project("SofaModeler")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(SOURCE_FILES

    Main.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaGUIMain SofaComponent )

include(${SOFA_CMAKE_DIR}/post.cmake)

