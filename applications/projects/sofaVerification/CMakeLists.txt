cmake_minimum_required(VERSION 2.8)

project("SofaVerification")

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES

    sofaVerification.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponentMain )

include(${SOFA_CMAKE_DIR}/postProject.cmake)

