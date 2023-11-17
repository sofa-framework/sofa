cmake_minimum_required(VERSION 3.12)

project(image_test)

set(HEADER_FILES
    TestImageEngine.h
)

set(SOURCE_FILES
    TestImageEngine.cpp
    DataImage_test.cpp
    ImageEngine_test.cpp
    ImageCImg_test.cpp
)
find_package(CImgPlugin REQUIRED)

add_definitions("-DIMAGETEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} image Sofa.Testing SceneCreator Sofa.Component.Engine.Testing CImgPlugin)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
