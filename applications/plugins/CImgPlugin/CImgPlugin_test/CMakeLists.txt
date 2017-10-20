cmake_minimum_required(VERSION 3.1)

project(CImgPlugin_test)

set(HEADER_FILES

)

set(SOURCE_FILES
    ImageCImg_test.cpp
)

add_definitions("-DCIMGPLUGIN_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/images\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} CImgPlugin SofaTest SofaGTestMain)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
