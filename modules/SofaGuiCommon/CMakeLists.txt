cmake_minimum_required(VERSION 3.12)
project(SofaGuiCommon)

sofa_find_package(SofaBase REQUIRED)
sofa_find_package(SofaUserInteraction REQUIRED)
sofa_find_package(SofaGraphComponent REQUIRED)
sofa_find_package(SofaSimulationCommon REQUIRED)
sofa_find_package(Sofa.GL QUIET) # ColourPickingVisitor
if(Sofa.GL_FOUND)
    message("-- ${PROJECT_NAME}: Sofa.GL dependent features enabled.")
else()
    message("-- ${PROJECT_NAME}: Sofa.GL dependent features disabled.")
endif()

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
target_link_libraries(${PROJECT_NAME} PUBLIC SofaBase SofaGraphComponent SofaUserInteraction SofaSimulationCommon)

# This is required to be able to #include <sofa/gui/BaseGui.h> instead of #include <SofaGuiCommon/sofa/gui/BaseGui.h>
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}/${PROJECT_NAME}>")

if(Sofa.GL_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL)
endif()

if (NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    # Silence attribute warnings (for example, ignored already defined external template)
    target_compile_options(${PROJECT_NAME} PRIVATE -Wno-attributes)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )

# sofa_create_package_with_targets add automatically PROJECT_NAME as include_dir
# but SofaGui's code is like sofa/gui... so we need to forcefully add src/SofaGuiCommon
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include/${PROJECT_NAME}/${PROJECT_NAME}>")
