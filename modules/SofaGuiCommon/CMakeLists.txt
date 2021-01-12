cmake_minimum_required(VERSION 3.12)
project(SofaGuiCommon)

find_package(SofaBase REQUIRED)
find_package(SofaUserInteraction REQUIRED)
find_package(SofaGraphComponent REQUIRED)
sofa_find_package(Sofa.GL) # ColourPickingVisitor

set(SRC_ROOT src/sofa/gui)

set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/BaseGUI.h
    ${SRC_ROOT}/BaseViewer.h
    ${SRC_ROOT}/BatchGUI.h
    ${SRC_ROOT}/ColourPickingVisitor.h
    ${SRC_ROOT}/FilesRecentlyOpenedManager.h
    ${SRC_ROOT}/GuiDataRepository.h
    ${SRC_ROOT}/GUIManager.h
    ${SRC_ROOT}/MouseOperations.h
    ${SRC_ROOT}/OperationFactory.h
    ${SRC_ROOT}/PickHandler.h
    ${SRC_ROOT}/SofaGUI.h
    ${SRC_ROOT}/ViewerFactory.h
)

set(SOURCE_FILES
    ${SRC_ROOT}/BaseGUI.cpp
    ${SRC_ROOT}/BaseViewer.cpp
    ${SRC_ROOT}/BatchGUI.cpp
    ${SRC_ROOT}/ColourPickingVisitor.cpp
    ${SRC_ROOT}/FilesRecentlyOpenedManager.cpp
    ${SRC_ROOT}/GuiDataRepository.cpp
    ${SRC_ROOT}/GUIManager.cpp
    ${SRC_ROOT}/MouseOperations.cpp
    ${SRC_ROOT}/PickHandler.cpp
    ${SRC_ROOT}/ViewerFactory.cpp
)

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBase SofaGraphComponent SofaUserInteraction)
if(Sofa.GL_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL)
endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_add_targets_to_package(
    PACKAGE_NAME SofaGui
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaGui"
    )
