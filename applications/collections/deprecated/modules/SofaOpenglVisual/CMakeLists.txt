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

#message(WARNING "${PROJECT_NAME} module is deprecated. It will be removed at v23.06. Use Sofa.GL.Component.Rendering2D, Sofa.GL.Component.Rendering3D and Sofa.GL.Component.Shader instead.")

# forward to the new sofang module
sofa_find_package(Sofa.GL.Component.Rendering2D REQUIRED)
sofa_find_package(Sofa.GL.Component.Rendering3D REQUIRED)
sofa_find_package(Sofa.GL.Component.Shader REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL.Component.Rendering2D)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL.Component.Rendering3D)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL.Component.Shader)

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaOpenglVisual"
    RELOCATABLE "collections"
    )
