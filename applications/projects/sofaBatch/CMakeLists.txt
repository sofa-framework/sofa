cmake_minimum_required(VERSION 2.8)

project("SofaBatch")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(SOURCE_FILES

    sofaBatch.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponentMain )

include(${SOFA_CMAKE_DIR}/post.cmake)

