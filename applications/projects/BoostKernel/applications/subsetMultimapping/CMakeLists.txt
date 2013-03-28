cmake_minimum_required(VERSION 2.8)

project("SubsetMultiMapping")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(SOURCE_FILES

    subsetMultimapping.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponentMain SofaObjectCreator SofaGUIMain )

include(${SOFA_CMAKE_DIR}/post.cmake)

