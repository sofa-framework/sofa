cmake_minimum_required(VERSION 2.8.12)
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

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaGraphComponent SofaBaseCollision SofaUserInteraction SofaBaseVisual SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUI")

install(TARGETS ${PROJECT_NAME}
        EXPORT SofaGuiTargets
        RUNTIME DESTINATION bin COMPONENT SofaGui_libraries
        LIBRARY DESTINATION lib COMPONENT SofaGui_libraries
        ARCHIVE DESTINATION lib COMPONENT SofaGui_libraries)

install(FILES ${HEADER_FILES} DESTINATION include/sofa/gui COMPONENT SofaGui_headers)
