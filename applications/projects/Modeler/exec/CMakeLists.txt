cmake_minimum_required(VERSION 3.22)
project(Modeler)


if(APPLE)
    set(RC_FILES "../Modeler.icns")
else()
    set(RC_FILES "sofa.rc")
endif()

sofa_find_package(Qt5 COMPONENTS Core REQUIRED)
sofa_find_package(SofaComponentAll REQUIRED)

if(Qt5Core_FOUND)
    message("Modeler executable: Using Qt5")
    qt5_add_resources(RESOURCE_FILES "${CMAKE_SOURCE_DIR}/share/textures/modeler/icons.qrc")
endif()

add_executable(${PROJECT_NAME} Main.cpp ${RC_FILES} ${RESOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaModeler SofaComponentAll SofaSimulationGraph)

sofa_install_targets(SofaModeler Modeler "Modeler")
