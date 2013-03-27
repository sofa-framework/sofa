cmake_minimum_required(VERSION 2.8)

project("CenterOfMassMulti2MappingChain")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(SOURCE_FILES

    centerOfMassMulti2MappingChain.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponentMain SofaObjectCreator SofaGuiMain )

include(${SOFA_CMAKE_DIR}/post.cmake)

