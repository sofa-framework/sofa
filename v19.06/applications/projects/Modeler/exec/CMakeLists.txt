cmake_minimum_required(VERSION 3.1)
project(Modeler)

if(APPLE)
    set(RC_FILES "../Modeler.icns")
else()
    set(RC_FILES "sofa.rc")
endif()

find_package(Qt5 COMPONENTS Core REQUIRED)

if(Qt5Core_FOUND)
    message("Modeler executable: Using Qt5")
    qt5_add_resources(RESOURCE_FILES "${CMAKE_SOURCE_DIR}/share/textures/modeler/icons.qrc")
endif()

add_executable(${PROJECT_NAME} Main.cpp ${RC_FILES} ${RESOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaModeler SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)

sofa_install_targets(SofaGui Modeler "")
