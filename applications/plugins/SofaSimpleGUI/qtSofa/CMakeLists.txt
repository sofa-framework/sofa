include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
	qtSofa.cpp
	QSofaViewer.cpp
        QSofaMainWindow.cpp
        QSofaScene.cpp
)
set(HEADER_FILES
	QSofaViewer.h
        QSofaMainWindow.h
        QSofaScene.h
)

UseQt()

SOFA_QT4_WRAP_CPP(MOC_FILES ${HEADER_FILES})

# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")

set(MOC_FILES_GROUP "Generated")
	
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES})

AddLinkerDependencies(SofaSimpleGUI ${GLUT_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)

