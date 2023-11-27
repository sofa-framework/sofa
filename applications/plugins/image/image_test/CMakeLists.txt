cmake_minimum_required(VERSION 3.12)

project(image_test)

set(HEADER_FILES
    TestImageEngine.h
)

set(SOURCE_FILES
    ImageCImg_test.cpp
)

if(NOT PLUGIN_IMAGE_COMPILE_SET STREQUAL "none")
    list(APPEND SOURCE_FILES
        TestImageEngine.cpp
        DataImage_test.cpp
        ImageEngine_test.cpp
    )

endif()
add_definitions("-DIMAGETEST_SCENES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/scenes\"")
add_definitions("-DIMAGETEST_RESOURCES_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/images\"")

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} image Sofa.Testing SceneCreator Sofa.Component.Engine.Testing)

add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})
