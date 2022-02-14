cmake_minimum_required(VERSION 3.12)
project(SofaOpenglVisual LANGUAGES CXX)

set(HEADER_FILES
    src/SofaOpenglVisual/config.h
    src/SofaOpenglVisual/initSofaOpenglVisual.h
    )
set(SOURCE_FILES
    src/SofaOpenglVisual/initSofaOpenglVisual.cpp
    )
set(EXTRA_FILES
    README.md
    )


add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

message(WARNING "${PROJECT_NAME} is deprecated. It will be removed at v23.06. Use Sofa.GL.Component.OnScreen, Sofa.GL.Component.Model and Sofa.GL.Component.Rendering instead.")

# forward to the new sofang module
sofa_find_package(Sofa.GL.Component.OnScreen REQUIRED)
sofa_find_package(Sofa.GL.Component.Model REQUIRED)
sofa_find_package(Sofa.GL.Component.Rendering REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL.Component.OnScreen)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL.Component.Model)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL.Component.Rendering)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaOpenglVisual"
    RELOCATABLE "plugins"
    )
