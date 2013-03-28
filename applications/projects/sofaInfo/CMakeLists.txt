cmake_minimum_required(VERSION 2.8)

project("SofaInfo")

include(${SOFA_CMAKE_DIR}/pre.cmake)



set(SOURCE_FILES

    sofaInfo.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaGuiMain )

include(${SOFA_CMAKE_DIR}/post.cmake)

