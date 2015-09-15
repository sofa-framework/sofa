cmake_minimum_required(VERSION 2.8.12)
project(SofaSimpleGUI)

set(SOFASIMPLEGUI_VERSION 0.1)

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


find_package(SofaMisc REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFASIMPLEGUI_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFASIMPLEGUI")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")


install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaSimpleGUI_libraries
        EXPORT SofaSimpleGUITargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})

## CMake package configurations files
include(SofaPackageConfigHelpers)
sofa_write_package_config_files(SofaSimpleGUI ${SOFASIMPLEGUI_VERSION} "" "include")
