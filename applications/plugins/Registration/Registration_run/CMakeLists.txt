cmake_minimum_required(VERSION 2.8)
project(Registration_run)

find_package(SofaSimpleGUI QUIET)
find_package(SofaPython QUIET)

if(SofaPython_FOUND AND SofaSimpleGUI_FOUND)

    find_package(Qt4 REQUIRED QtCore)
    include(${QT_USE_FILE})

    set(SOURCE_FILES
	    Registration_run.cpp
        QSofaScene.cpp
        QSofaViewer.cpp
        QMainWindow_RegistrationRun.cpp)

    set(HEADER_FILES
        QSofaScene.h
        QSofaViewer.h
        QMainWindow_RegistrationRun.h)

    qt4_wrap_cpp(MOC_FILES ${HEADER_FILES})

    qt4_add_resources(RESOURCE_FILES "./icons.qrc")

    add_definitions("-Dregistration_SRC_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")

    add_executable(${PROJECT_NAME} ${HEADER_FILES} ${MOC_FILES} ${SOURCE_FILES} ${RESOURCE_FILES})
    target_include_directories(${PROJECT_NAME} PRIVATE ${QT_INCLUDES})
    target_link_libraries(${PROJECT_NAME} SofaPython SofaSimpleGUI ${QT_LIBRARIES} ${GLUT_LIBRARIES})

endif()
