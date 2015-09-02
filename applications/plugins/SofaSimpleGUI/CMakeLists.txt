cmake_minimum_required(VERSION 2.8.12)
project(SofaSimpleGUI)

set(SOFASIMPLEGUI_MAJOR_VERSION 0)
set(SOFASIMPLEGUI_MINOR_VERSION 0)
set(SOFASIMPLEGUI_BUILD_VERSION 1)

set(SOFASIMPLEGUI_VERSION ${SOFASIMPLEGUI_MAJOR_VERSION}.${SOFASIMPLEGUI_MINOR_VERSION}.${SOFASIMPLEGUI_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")


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


# SofaSimpleGUIConfig.cmake (build tree)
set(SOFASIMPLEGUI_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
configure_package_config_file(SofaSimpleGUIConfig.cmake.in
                              SofaSimpleGUIConfig.cmake
                              INSTALL_DESTINATION "${CMAKE_CURRENT_BINARY_DIR}"
                              PATH_VARS SOFASIMPLEGUI_INCLUDE_DIR)

# SofaSimpleGUIConfig.cmake (installed)
set(SOFASIMPLEGUI_INCLUDE_DIR "include")
configure_package_config_file(SofaSimpleGUIConfig.cmake.in
                              SofaSimpleGUIConfigInstall.cmake
                              INSTALL_DESTINATION lib/cmake/SofaSimpleGUI
                              PATH_VARS SOFASIMPLEGUI_INCLUDE_DIR)

# SofaSimpleGUITargets.cmake (installed)
install(EXPORT SofaSimpleGUITargets
        DESTINATION lib/cmake/SofaSimpleGUI)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SofaSimpleGUIConfigInstall.cmake"
        DESTINATION lib/cmake/SofaSimpleGUI
        RENAME SofaSimpleGUIConfig.cmake)

install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaSimpleGUI_libraries
        EXPORT SofaSimpleGUITargets
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        PUBLIC_HEADER DESTINATION "include/${PROJECT_NAME}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
