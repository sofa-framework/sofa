cmake_minimum_required(VERSION 3.12)
project(SofaSimpleGUI)

set(SOFASIMPLEGUI_VERSION 0.1)

if(${SOFA_NO_OPENGL})
    message(SEND_ERROR "Warning: CMake flag SOFA_NO_OPENGL is active.")
    message("Flag SOFA_NO_OPENGL incompatible with the SofaSimpleGUI plugin.")
endif()

set(HEADER_FILES
    Camera.h
    Interactor.h
    PickedPoint.h
    SofaGL.h
    SofaScene.h
    SpringInteractor.h
    VisualPickVisitor.h
    config.h
)

set(SOURCE_FILES
    Camera.cpp
    Interactor.cpp
    SofaGL.cpp
    SofaScene.cpp
    SpringInteractor.cpp
    VisualPickVisitor.cpp
    initSimpleGUI.cpp
)

find_package(SofaGeneral REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaSimulationGraph SofaBase SofaCommon SofaGeneral)
target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFASIMPLEGUI_VERSION})

## Install rules for the library and headers; CMake package configurations files
sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${SOFASIMPLEGUI_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )
