cmake_minimum_required(VERSION 3.1)

project(SofaGeneralLoader_test)

set(SOURCE_FILES 
    # MeshSTLLoader_test.cpp
    ReadState_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaTest)
add_definitions("-DSOFAGENERALLOADER_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/files/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
