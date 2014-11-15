include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(SOURCE_FILES
	    Registration_run.cpp
        QSofaScene.cpp
        QSofaViewer.cpp
        QMainWindow_RegistrationRun.cpp
)
set(HEADER_FILES
        QSofaScene.h
        QSofaViewer.h
        QMainWindow_RegistrationRun.h
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
	
AddCompilerDefinitions("registration_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

add_executable(${PROJECT_NAME} ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${RESOURCE_FILES})

AddLinkerDependencies(SofaPython SofaSimpleGUI ${GLUT_LIBRARIES})

include(${SOFA_CMAKE_DIR}/postProject.cmake)

