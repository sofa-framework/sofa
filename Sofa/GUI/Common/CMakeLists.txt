cmake_minimum_required(VERSION 3.12)
project(Sofa.GUI.Common LANGUAGES CXX)

find_package(cxxopts 3.1 QUIET)
if(NOT cxxopts_FOUND AND SOFA_ALLOW_FETCH_DEPENDENCIES)
    message("Sofa.GUI.Common: DEPENDENCY cxxopt NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is ON, fetching cxxopt...")

    include(FetchContent)
    FetchContent_Declare(cxxopts
            GIT_REPOSITORY https://github.com/jarro2783/cxxopts
            GIT_TAG        v3.1.1
    )

    FetchContent_GetProperties(cxxopts)
    if(NOT cxxopts_POPULATED)
        FetchContent_Populate(cxxopts)

        set(CXXOPTS_BUILD_EXAMPLES OFF CACHE INTERNAL "")
        set(CXXOPTS_BUILD_TESTS OFF CACHE INTERNAL "")
        set(CXXOPTS_ENABLE_INSTALL ON CACHE INTERNAL "")
        message("Sofa.GUI.Common: adding subdirectory ${cxxopts_SOURCE_DIR}")

        add_subdirectory(${cxxopts_SOURCE_DIR} ${cxxopts_BINARY_DIR})
    endif()
elseif (NOT cxxopts_FOUND)
    message(FATAL_ERROR "Sofa.GUI.Common: DEPENDENCY cxxopt NOT FOUND. SOFA_ALLOW_FETCH_DEPENDENCIES is OFF and thus cannot be fetched. Install libcxxopts-dev (version>=3.1.0), or enable SOFA_ALLOW_FETCH_DEPENDENCIES to fix this issue.")
endif()

set(SOFAGUICOMMON_ROOT src/sofa/gui/common)

set(HEADER_FILES
    ${SOFAGUICOMMON_ROOT}/config.h.in
    ${SOFAGUICOMMON_ROOT}/init.h
    ${SOFAGUICOMMON_ROOT}/BaseGUI.h
    ${SOFAGUICOMMON_ROOT}/BaseViewer.h
    ${SOFAGUICOMMON_ROOT}/ColourPickingVisitor.h
    ${SOFAGUICOMMON_ROOT}/FilesRecentlyOpenedManager.h
    ${SOFAGUICOMMON_ROOT}/GuiDataRepository.h
    ${SOFAGUICOMMON_ROOT}/GUIManager.h
    ${SOFAGUICOMMON_ROOT}/MouseOperations.h
    ${SOFAGUICOMMON_ROOT}/OperationFactory.h
    ${SOFAGUICOMMON_ROOT}/PickHandler.h
    ${SOFAGUICOMMON_ROOT}/ViewerFactory.h
    ${SOFAGUICOMMON_ROOT}/ArgumentParser.h
)

set(SOURCE_FILES
    ${SOFAGUICOMMON_ROOT}/init.cpp
    ${SOFAGUICOMMON_ROOT}/BaseGUI.cpp
    ${SOFAGUICOMMON_ROOT}/BaseViewer.cpp
    ${SOFAGUICOMMON_ROOT}/ColourPickingVisitor.cpp
    ${SOFAGUICOMMON_ROOT}/FilesRecentlyOpenedManager.cpp
    ${SOFAGUICOMMON_ROOT}/GuiDataRepository.cpp
    ${SOFAGUICOMMON_ROOT}/GUIManager.cpp
    ${SOFAGUICOMMON_ROOT}/MouseOperations.cpp
    ${SOFAGUICOMMON_ROOT}/PickHandler.cpp
    ${SOFAGUICOMMON_ROOT}/ViewerFactory.cpp
    ${SOFAGUICOMMON_ROOT}/ArgumentParser.cpp
)

# Compatibility files
set (COMPATSOFAGUICOMMON_ROOT "compat/")
set (COMPAT_HEADER_FILES
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/ArgumentParser.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/BaseGUI.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/BatchGUI.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/BaseViewer.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/ColourPickingVisitor.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/FilesRecentlyOpenedManager.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/GuiDataRepository.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/GUIManager.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/MouseOperations.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/OperationFactory.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/PickHandler.h
    ${COMPATSOFAGUICOMMON_ROOT}/sofa/gui/ViewerFactory.h
)

sofa_find_package(Sofa.Simulation.Common REQUIRED)
sofa_find_package(Sofa.Component.Setting REQUIRED)
sofa_find_package(Sofa.Component.Collision.Response.Contact REQUIRED)
sofa_find_package(Sofa.GUI.Component REQUIRED)
sofa_find_package(Sofa.GL QUIET) # ColourPickingVisitor

if(Sofa.GL_FOUND)
    message(STATUS "-- ${PROJECT_NAME}: Sofa.GL dependent features enabled.")
else()
    message(WARNING "-- ${PROJECT_NAME}: Sofa.GL dependent features disabled.")
endif()

add_library(${PROJECT_NAME} SHARED ${HEADER_FILES} ${SOURCE_FILES} ${COMPAT_HEADER_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Simulation.Common)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Setting)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.Component.Collision.Response.Contact)
target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GUI.Component)
target_link_libraries(${PROJECT_NAME} PUBLIC cxxopts::cxxopts)

target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${COMPATSOFAGUICOMMON_ROOT}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include/${PROJECT_NAME}/${PROJECT_NAME}/compat>")
source_group("compat" FILES ${COMPAT_HEADER_FILES} )

if(Sofa.GL_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC Sofa.GL)
endif()

sofa_create_package_with_targets(
    PACKAGE_NAME ${PROJECT_NAME}
    PACKAGE_VERSION ${Sofa_VERSION}
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR ${PROJECT_NAME}
    )
