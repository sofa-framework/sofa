cmake_minimum_required(VERSION 2.8)
project(qtSofa)

set(SOURCE_FILES
	qtSofa.cpp
	QSofaMainWindow.cpp
	QSofaScene.cpp
	QSofaViewer.cpp)

set(HEADER_FILES
	QSofaMainWindow.h
	QSofaScene.h
    QSofaViewer.h
    oneTetra.h)

find_package(SofaSimpleGUI)
find_package(GLUT)

# UseQt()

qt4_wrap_cpp(MOC_FILES ${HEADER_FILES})

# # make resource files path from absolute to relative
# ToRelativePath(RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${RESOURCE_FILES}")

set(RESOURCE_FILES_GROUP "Generated")

qt4_add_resources(RESOURCE_FILES "./icons.qrc")

# # make mocced header / form files path from absolute to relative
# ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

set(MOC_FILES_GROUP "Generated")

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${RESOURCE_FILES})
target_link_libraries(${PROJECT_NAME} SofaSimpleGUI ${GLUT_LIBRARIES})
compile_definitions("-DQTSOFA_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
