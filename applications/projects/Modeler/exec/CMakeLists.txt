cmake_minimum_required(VERSION 2.8.12)
project(Modeler)

if(APPLE)
    set(RC_FILES "../Modeler.icns")
else()
    set(RC_FILES "sofa.rc")
endif()

# TODO QT5
find_package(Qt4)
qt4_add_resources(RESOURCE_FILES "${CMAKE_SOURCE_DIR}/share/textures/modeler/icons.qrc")
set(RESOURCE_FILES_GROUP "Generated")

add_executable(${PROJECT_NAME} Main.cpp ${RC_FILES} ${RESOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaModeler SofaComponentBase SofaComponentCommon SofaComponentGeneral SofaComponentAdvanced SofaComponentMisc)

sofa_install_targets(SofaGui Modeler "")
