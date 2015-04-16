cmake_minimum_required(VERSION 2.8.12)
project(SofaSimpleGUI)

set(SOFASIMPLEGUI_MAJOR_VERSION 0)
set(SOFASIMPLEGUI_MINOR_VERSION 0)
set(SOFASIMPLEGUI_BUILD_VERSION 1)

set(SOFASIMPLEGUI_VERSION ${SOFASIMPLEGUI_MAJOR_VERSION}.${SOFASIMPLEGUI_MINOR_VERSION}.${SOFASIMPLEGUI_BUILD_VERSION} CACHE INTERNAL "Set to internal so it can be changed by sub directories.")

set(SOFASIMPLEGUI_INSTALL_LIB_DIR     lib                      CACHE PATH "Installation directory for libraries relative to CMAKE_INSTALL_PREFIX variable")
set(SOFASIMPLEGUI_INSTALL_BIN_DIR     bin                      CACHE PATH "Installation directory for executables relative to CMAKE_INSTALL_PREFIX variable")
set(SOFASIMPLEGUI_INSTALL_INCLUDE_DIR include                  CACHE PATH "Installation directory for header files relative to CMAKE_INSTALL_PREFIX variable")
set(SOFASIMPLEGUI_INSTALL_CMAKE_DIR   ${SOFASIMPLEGUI_INSTALL_LIB_DIR}/cmake/SofaSimpleGUI )

set(HEADER_FILES
    initSimpleGUI.h
    Camera.h
    Interactor.h
    SpringInteractor.h
    PickedPoint.h
    SofaScene.h
    SofaGL.h
    VisualPickVisitor.h)

set(SOURCE_FILES
    initSimpleGUI.cpp
    Camera.cpp
    Interactor.cpp
    SpringInteractor.cpp
    SofaScene.cpp
    SofaGL.cpp
    VisualPickVisitor.cpp)

find_package(SofaGeneral REQUIRED)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimulationGraph SofaComponentGeneral)

set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${SOFASIMPLEGUI_VERSION})
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFASIMPLEGUI")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")


install(TARGETS ${PROJECT_NAME}
        COMPONENT SofaSimpleGUI_libraries
        EXPORT  SofaSimpleGUITargets
        RUNTIME DESTINATION "${SOFASIMPLEGUI_INSTALL_BIN_DIR}"
        LIBRARY DESTINATION "${SOFASIMPLEGUI_INSTALL_LIB_DIR}"
        ARCHIVE DESTINATION "${SOFASIMPLEGUI_INSTALL_LIB_DIR}"
        PUBLIC_HEADER DESTINATION "${SOFASIMPLEGUI_INSTALL_INCLUDE_DIR}/${PROJECT_NAME}")

install(DIRECTORY examples/ DESTINATION share/sofa/plugins/${PROJECT_NAME})
