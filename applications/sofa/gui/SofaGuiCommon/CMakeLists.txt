cmake_minimum_required(VERSION 3.1)
project(SofaGuiCommon)

find_package(SofaBase REQUIRED)
find_package(SofaCommon REQUIRED)
find_package(SofaMisc REQUIRED)
find_package(SofaGeneral REQUIRED)

set(HEADER_FILES
    ../BaseGUI.h
    ../BaseViewer.h
    ../BatchGUI.h
    ../ColourPickingVisitor.h
    ../FilesRecentlyOpenedManager.h
    ../GuiDataRepository.h
    ../GUIManager.h
    ../MouseOperations.h
    ../OperationFactory.h
    ../PickHandler.h
    ../SofaGUI.h
    ../ViewerFactory.h
)

set(SOURCE_FILES
    ../BaseGUI.cpp
    ../BaseViewer.cpp
    ../BatchGUI.cpp
    ../ColourPickingVisitor.cpp
    ../FilesRecentlyOpenedManager.cpp
    ../GuiDataRepository.cpp
    ../GUIManager.cpp
    ../MouseOperations.cpp
    ../PickHandler.cpp
    ../ViewerFactory.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBase SofaCommon SofaGeneral SofaMisc)
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUI")
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${HEADER_FILES}")

sofa_install_targets(SofaGui SofaGuiCommon "SofaGui/sofa/gui")
