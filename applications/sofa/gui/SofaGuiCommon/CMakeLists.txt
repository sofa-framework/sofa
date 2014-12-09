cmake_minimum_required(VERSION 2.8)
project(SofaGuiCommon)

set(HEADER_FILES
    ../BaseGUI.h
    ../BaseViewer.h
    ../BatchGUI.h
    ../ColourPickingVisitor.h
    ../MouseOperations.h
    ../OperationFactory.h
    ../PickHandler.h
    ../FilesRecentlyOpenedManager.h
    ../SofaGUI.h
    ../ViewerFactory.h
    ../GUIManager.h
    )

set(SOURCE_FILES
    ../BaseGUI.cpp
    ../BaseViewer.cpp
    ../BatchGUI.cpp
    ../ColourPickingVisitor.cpp
    ../FilesRecentlyOpenedManager.cpp
    ../MouseOperations.cpp
    ../PickHandler.cpp
    ../GUIManager.cpp
    ../ViewerFactory.cpp
    )

set(GROUP_BASE_DIR "..")

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})

target_link_libraries( ${PROJECT_NAME} SofaGraphComponent SofaBaseCollision SofaUserInteraction SofaBaseVisual SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)

set( SOFAGUICOMMON_COMPILER_FLAGS "-DSOFA_BUILD_SOFAGUI")


set_target_properties( ${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "_d" )
set_target_properties( ${PROJECT_NAME} PROPERTIES VERSION ${SOFAGUI_VERSION} )
set_target_properties( ${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${SOFAGUICOMMON_COMPILER_FLAGS}"  )
# set_target_properties( ${PROJECT_NAME} PROPERTIES PUBLIC_HEADER ${HEADER_FILES} )

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../../>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGUI_BUILD_INCLUDE_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${SOFAGUI_DEPENDENCY_DIR}>")

install(TARGETS ${PROJECT_NAME}
        EXPORT  SofaGuiTargets
        RUNTIME DESTINATION "${SOFAGUI_INSTALL_BIN_DIR}" COMPONENT SofaGui_libraries
        LIBRARY DESTINATION "${SOFAGUI_INSTALL_LIB_DIR}" COMPONENT SofaGui_libraries
        ARCHIVE DESTINATION "${SOFAGUI_INSTALL_LIB_DIR}" COMPONENT SofaGui_libraries
#        PUBLIC_HEADER DESTINATION "${SOFAGUI_INSTALL_INCLUDE_DIR}/sofa/gui" COMPONENT SofaGui_headers
        )
        
install(FILES ${HEADER_FILES} DESTINATION "${SOFAGUI_INSTALL_INCLUDE_DIR}/sofa/gui"
        COMPONENT SofaGui_headers
        )
