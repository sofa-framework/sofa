cmake_minimum_required(VERSION 2.8)
project(qtSofa)

set(SOURCE_FILES
	QSofaMainWindow.cpp
	QSofaScene.cpp
	QSofaViewer.cpp
	qtSofa.cpp
)

set(HEADER_FILES
	QSofaMainWindow.h
	QSofaScene.h
    QSofaViewer.h
    oneTetra.h
)

find_package(SofaSimpleGUI REQUIRED)
find_package(GLUT REQUIRED)

qt4_wrap_cpp(MOC_FILES ${HEADER_FILES})
set(RESOURCE_FILES_GROUP "Generated")

qt4_add_resources(RESOURCE_FILES "./icons.qrc")
set(MOC_FILES_GROUP "Generated")

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${RESOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimpleGUI ${GLUT_LIBRARIES})
compile_definitions("-DQTSOFA_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
