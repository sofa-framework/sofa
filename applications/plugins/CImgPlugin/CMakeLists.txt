cmake_minimum_required(VERSION 3.12)
project(CImgPlugin VERSION 0.1 LANGUAGES CXX)

message(WARNING "CImgPlugin has been merged in the image plugin. If you need cimg feature, please use the image plugin instead.")

find_package(image REQUIRED)

set(SOURCE_FILES
    initCImgPlugin.cpp
    )

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC image)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${PROJECT_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "CImgPlugin"
    RELOCATABLE "plugins"
    )
