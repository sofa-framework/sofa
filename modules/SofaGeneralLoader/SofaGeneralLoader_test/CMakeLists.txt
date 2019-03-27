cmake_minimum_required(VERSION 3.1)

project(SofaGeneralLoader_test)

set(SOURCE_FILES 
    MeshXspLoader_test.cpp
    ReadState_test.cpp
    )

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGTestMain SofaGeneralLoader)
target_compile_definitions(${PROJECT_NAME} PRIVATE "SOFAGENERALLOADER_TESTFILES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/files/\"")

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
