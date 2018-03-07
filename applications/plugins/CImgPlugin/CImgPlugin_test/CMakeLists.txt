cmake_minimum_required(VERSION 3.1)

project(CImgPlugin_test)

find_package(CImgPlugin REQUIRED)

set(SOURCE_FILES
    ImageCImg_test.cpp
)

add_definitions("-DCIMGPLUGIN_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/images\"")
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
target_link_libraries(${PROJECT_NAME} CImgPlugin SofaCore SofaGTestMain)
target_include_directories(${PROJECT_NAME} PRIVATE ${CImgPlugin_INCLUDE_DIRS})


