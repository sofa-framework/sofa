cmake_minimum_required(VERSION 2.8)

project("CenterOfMassMapping")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(SOURCE_FILES

    centerOfMassMapping.cpp

    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

set(LINKER_DEPENDENCIES SofaComponent SofaObjectCreator SofaGuiMain )

include(${SOFA_CMAKE_DIR}/post.cmake)

