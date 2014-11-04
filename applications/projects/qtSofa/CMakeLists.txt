include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
	qtSofa.cpp
	QSofaMainWindow.cpp
	QSofaScene.cpp
	QSofaViewer.cpp
)
set(HEADER_FILES
	QSofaMainWindow.h
	QSofaScene.h
    QSofaViewer.h
    oneTetra.h
)

UseQt()

SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})

# make resource files path from absolute to relative
ToRelativePath(RESOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${RESOURCE_FILES}")

set(RESOURCE_FILES_GROUP "Generated")

QT4_ADD_RESOURCES(RESOURCE_FILES "./icons.qrc")

# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

set(MOC_FILES_GROUP "Generated")
	
include_directories(${SOFA_EXTLIBS_DIR})

AddCompilerDefinitions("QTSOFA_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${RESOURCE_FILES})

AddLinkerDependencies(SofaSimpleGUI ${GLUT_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)

