cmake_minimum_required(VERSION 3.12)
project(SofaGuiQt)

# Qt dependencies
set(QT_TARGETS "")
set(QT_USE_IMPORTED_TARGETS 1)
set(QT5_NO_LINK_QTMAIN 1)
sofa_find_package(Qt5 COMPONENTS Core Gui OpenGL QUIET)

if(NOT Qt5Core_FOUND)
    message(SEND_ERROR "SofaGuiQt: Qt5 is required and was not found.\n"
        "Add Qt5 directory to CMAKE_PREFIX_PATH or disable SofaGuiQt by setting SOFAGUI_QT to OFF.")
endif()
set(QT_TARGETS ${QT_TARGETS} Qt5::Core Qt5::Gui Qt5::OpenGL)

sofa_find_package(Qt5 COMPONENTS Charts QUIET BOTH_SCOPES)
if(Qt5Charts_FOUND)
    set(QT_TARGETS ${QT_TARGETS} Qt5::Charts)
else()
    message(STATUS "SofaGuiQt: Qt5Charts not found. No chart nor Profile will be compiled")
endif()

# QDocBrowser
find_package(Qt5 COMPONENTS WebEngine QUIET) # if found, then QDocBrowser will be ON by default
option(SOFAGUIQT_ENABLE_QDOCBROWSER "Build the QDocBrowser. Qt5WebEngine is needed." ${Qt5WebEngine_FOUND})
if(SOFAGUIQT_ENABLE_QDOCBROWSER)
    sofa_find_package(Qt5 COMPONENTS WebEngine WebEngineWidgets REQUIRED BOTH_SCOPES)
    set(QT_TARGETS ${QT_TARGETS} Qt5::WebEngine Qt5::WebEngineWidgets)
endif()

# QtViewer and QGLViewer
if(NOT SOFA_NO_OPENGL)
    # QtViewer
    # TODO: rename SOFAGUI_QTVIEWER into SOFAGUIQT_ENABLE_QTVIEWER
    option(SOFAGUI_QTVIEWER "Compile the QtViewer for the Qt GUI" ON)

    # QGLViewer
    # TODO: rename SOFAGUI_QGLVIEWER into SOFAGUIQT_ENABLE_QGLVIEWER
    option(SOFAGUI_QGLVIEWER "Compile the QGLViewer for the Qt GUI" ON)
    if(SOFAGUI_QGLVIEWER)
        find_package(QGLViewer QUIET)
        if(NOT QGLViewer_FOUND)
            add_subdirectory(libQGLViewer-2.7.1/QGLViewer)
        endif()
    endif()
else()
    set(SOFAGUI_QTVIEWER OFF)
    set(SOFAGUI_QGLVIEWER OFF)
endif()
sofa_set_01(SOFAGUIQT_HAVE_QTVIEWER VALUE ${SOFAGUI_QTVIEWER} BOTH_SCOPES)
sofa_set_01(SOFAGUIQT_HAVE_QGLVIEWER VALUE ${SOFAGUI_QGLVIEWER} BOTH_SCOPES)

# VSync
# TODO: rename SOFAGUI_ENABLE_VSYNC into SOFAGUIQT_ENABLE_VSYNC
option(SOFAGUI_ENABLE_VSYNC "Enable vertical refresh sync" OFF)
sofa_set_01(SOFAGUIQT_ENABLE_VSYNC_ VALUE ${SOFAGUI_ENABLE_VSYNC}) # see SofaGuiQt.h.in

# Qt native menu
# TODO: rename SOFAGUI_ENABLE_NATIVE_MENU into SOFAGUIQT_ENABLE_NATIVE_MENU
set(DEFAULT_SOFAGUI_ENABLE_NATIVE_MENU ON)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(DEFAULT_SOFAGUI_ENABLE_NATIVE_MENU OFF)
endif()
option(SOFAGUI_ENABLE_NATIVE_MENU "Enable Qt to use the operating system's native menu for the Qt GUI" ${DEFAULT_SOFAGUI_ENABLE_NATIVE_MENU})

# NodeEditor
option(SOFAGUIQT_ENABLE_NODEGRAPH "Build the Node graph representation. NodeEditor library is needed." OFF)
if(SOFAGUIQT_ENABLE_NODEGRAPH)
    sofa_find_package(NodeEditor REQUIRED BOTH_SCOPES)
endif()

# FFMPEG_exec
sofa_find_package(FFMPEG_exec BOTH_SCOPES)

# TinyXML
if(SOFA_DUMP_VISITOR_INFO)
    sofa_find_package(TinyXML REQUIRED BOTH_SCOPES)
endif()

set(SRC_ROOT src/sofa/gui/qt)

set(MOC_HEADER_FILES
    ${SRC_ROOT}/AddObject.h
    ${SRC_ROOT}/DataFilenameWidget.h
    ${SRC_ROOT}/DataWidget.h
    ${SRC_ROOT}/DisplayFlagsDataWidget.h
    ${SRC_ROOT}/GenGraphForm.h
    ${SRC_ROOT}/LinkWidget.h
    ${SRC_ROOT}/MaterialDataWidget.h
    ${SRC_ROOT}/RGBAColorDataWidget.h
    ${SRC_ROOT}/ModifyObject.h
    ${SRC_ROOT}/QDataDescriptionWidget.h
    ${SRC_ROOT}/QDisplayDataWidget.h
    ${SRC_ROOT}/QDisplayLinkWidget.h
    ${SRC_ROOT}/QDisplayPropertyWidget.h
    ${SRC_ROOT}/QRGBAColorPicker.h
    ${SRC_ROOT}/QModelViewTableUpdater.h
    ${SRC_ROOT}/QMouseOperations.h
    ${SRC_ROOT}/QMouseWheelAdjustementGuard.h
    ${SRC_ROOT}/QSofaListView.h
    ${SRC_ROOT}/QSofaStatWidget.h
    ${SRC_ROOT}/QTabulationModifyObject.h
    ${SRC_ROOT}/QTransformationWidget.h
    ${SRC_ROOT}/RealGUI.h
    ${SRC_ROOT}/SimpleDataWidget.h
    ${SRC_ROOT}/SofaMouseManager.h    
    ${SRC_ROOT}/SofaVideoRecorderManager.h
    ${SRC_ROOT}/SofaPluginManager.h
    ${SRC_ROOT}/WDoubleLineEdit.h    
    )
set(HEADER_FILES
    ${SRC_ROOT}/config.h.in
    ${SRC_ROOT}/SofaGuiQt.h
    ${SRC_ROOT}/qt.conf.h
    ${SRC_ROOT}/FileManagement.h
    ${SRC_ROOT}/GraphListenerQListView.h
    ${SRC_ROOT}/PickHandlerCallBacks.h
    ${SRC_ROOT}/QMenuFilesRecentlyOpened.h
    ${SRC_ROOT}/QModelViewTableDataContainer.h
    ${SRC_ROOT}/StructDataWidget.h
    ${SRC_ROOT}/TableDataWidget.h
    ${SRC_ROOT}/viewer/SofaViewer.h
    ${SRC_ROOT}/viewer/EngineBackend.h
    ${SRC_ROOT}/viewer/VisualModelPolicy.h
    ${SRC_ROOT}/GenericWidget.h
    )
set(SOURCE_FILES
    ${SRC_ROOT}/AddObject.cpp
    ${SRC_ROOT}/DataFilenameWidget.cpp
    ${SRC_ROOT}/DataWidget.cpp
    ${SRC_ROOT}/DisplayFlagsDataWidget.cpp
    ${SRC_ROOT}/FileManagement.cpp
    ${SRC_ROOT}/GenGraphForm.cpp
    ${SRC_ROOT}/GraphListenerQListView.cpp
    ${SRC_ROOT}/LinkWidget.cpp
    ${SRC_ROOT}/MaterialDataWidget.cpp
    ${SRC_ROOT}/RGBAColorDataWidget.cpp
    ${SRC_ROOT}/ModifyObject.cpp
    ${SRC_ROOT}/PickHandlerCallBacks.cpp
    ${SRC_ROOT}/QDataDescriptionWidget.cpp
    ${SRC_ROOT}/QDisplayDataWidget.cpp
    ${SRC_ROOT}/QDisplayLinkWidget.cpp
    ${SRC_ROOT}/QDisplayPropertyWidget.cpp
    ${SRC_ROOT}/QRGBAColorPicker.cpp
    ${SRC_ROOT}/QMenuFilesRecentlyOpened.cpp
    ${SRC_ROOT}/QModelViewTableUpdater.cpp
    ${SRC_ROOT}/QMouseOperations.cpp
    ${SRC_ROOT}/QMouseWheelAdjustementGuard.cpp
    ${SRC_ROOT}/QSofaListView.cpp
    ${SRC_ROOT}/QSofaStatWidget.cpp
    ${SRC_ROOT}/QTabulationModifyObject.cpp
    ${SRC_ROOT}/QTransformationWidget.cpp
    ${SRC_ROOT}/RealGUI.cpp
    ${SRC_ROOT}/SimpleDataWidget.cpp
    ${SRC_ROOT}/SofaMouseManager.cpp
    ${SRC_ROOT}/SofaPluginManager.cpp
    ${SRC_ROOT}/SofaVideoRecorderManager.cpp    
    ${SRC_ROOT}/StructDataWidget.cpp
    ${SRC_ROOT}/TableDataWidget.cpp
    ${SRC_ROOT}/WDoubleLineEdit.cpp
    ${SRC_ROOT}/viewer/SofaViewer.cpp
    ${SRC_ROOT}/viewer/VisualModelPolicy.cpp    
)
set(UI_FILES
    ${SRC_ROOT}/BaseGenGraphForm.ui
    ${SRC_ROOT}/DialogAddObject.ui
    ${SRC_ROOT}/GUI.ui
    ${SRC_ROOT}/MouseManager.ui
    ${SRC_ROOT}/PluginManager.ui
    ${SRC_ROOT}/VideoRecorderManager.ui    
)

if(SOFA_DUMP_VISITOR_INFO)
    list(APPEND MOC_HEADER_FILES
        ${SRC_ROOT}/WindowVisitor.h
        ${SRC_ROOT}/QVisitorControlPanel.h
        ${SRC_ROOT}/PieWidget.h
        )
    list(APPEND HEADER_FILES
        ${SRC_ROOT}/GraphVisitor.h
        )
    list(APPEND SOURCE_FILES
        ${SRC_ROOT}/GraphVisitor.cpp
        ${SRC_ROOT}/WindowVisitor.cpp
        ${SRC_ROOT}/QVisitorControlPanel.cpp
        ${SRC_ROOT}/PieWidget.cpp
        )
    list(APPEND UI_FILES
        ${SRC_ROOT}/VisitorGUI.ui
        )
endif()

if(NOT SOFA_NO_OPENGL)
    list(APPEND HEADER_FILES
        ${SRC_ROOT}/GLPickHandler.h
        ${SRC_ROOT}/viewer/GLBackend.h
        ${SRC_ROOT}/viewer/OglModelPolicy.h
        )
    list(APPEND SOURCE_FILES
        ${SRC_ROOT}/GLPickHandler.cpp
        ${SRC_ROOT}/viewer/GLBackend.cpp
        ${SRC_ROOT}/viewer/OglModelPolicy.cpp
        )

    # QtViewer
    if(SOFAGUI_QTVIEWER)
        list(APPEND MOC_HEADER_FILES ${SRC_ROOT}/viewer/qt/QtViewer.h)
        list(APPEND SOURCE_FILES ${SRC_ROOT}/viewer/qt/QtViewer.cpp)
    endif()

    # QGLViewer
    if(SOFAGUI_QGLVIEWER)
        list(APPEND MOC_HEADER_FILES ${SRC_ROOT}/viewer/qgl/QtGLViewer.h)
        list(APPEND SOURCE_FILES ${SRC_ROOT}/viewer/qgl/QtGLViewer.cpp)
    endif()
endif()

# QtCharts
if(Qt5Charts_FOUND)
    list(APPEND MOC_HEADER_FILES 
        ${SRC_ROOT}/SofaWindowProfiler.h
        ${SRC_ROOT}/QGraphStatWidget.h
        ${SRC_ROOT}/QEnergyStatWidget.h
        ${SRC_ROOT}/QMomentumStatWidget.h
        ${SRC_ROOT}/GraphDataWidget.h
        )
    list(APPEND SOURCE_FILES 
        ${SRC_ROOT}/SofaWindowProfiler.cpp
        ${SRC_ROOT}/QGraphStatWidget.cpp
        ${SRC_ROOT}/QEnergyStatWidget.cpp
        ${SRC_ROOT}/QMomentumStatWidget.cpp
        ${SRC_ROOT}/GraphDataWidget.cpp
        )
    list(APPEND UI_FILES ${SRC_ROOT}/WindowProfiler.ui)
else()
    message(STATUS "SofaGuiQt: QtCharts not found, SofaWindowProfiler will not be built.")
endif()

if(SOFAGUIQT_ENABLE_QDOCBROWSER)
    list(APPEND MOC_HEADER_FILES ${SRC_ROOT}/panels/QDocBrowser.h)
    list(APPEND SOURCE_FILES ${SRC_ROOT}/panels/QDocBrowser.cpp)
else()
    message(STATUS "SofaGuiQt: Qt5WebEngine not found, QDocBrowser will not be built.")
endif()


#NodeEditor
if (SOFAGUIQT_ENABLE_NODEGRAPH)
    list(APPEND MOC_HEADER_FILES 
        ${SRC_ROOT}/SofaWindowDataGraph.h
        ${SRC_ROOT}/dataGraph/SofaComponentNodeModel.h
        )

    list(APPEND SOURCE_FILES 
        ${SRC_ROOT}/SofaWindowDataGraph.cpp
        ${SRC_ROOT}/dataGraph/SofaComponentNodeModel.cpp
        )
endif()

qt5_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
set(MOC_FILES_GROUP "Generated")
qt5_wrap_ui(FORM_FILES ${UI_FILES})
set(FORM_FILES_GROUP "Generated")

find_package(SofaMiscForceField REQUIRED)

add_library(${PROJECT_NAME} SHARED ${MOC_HEADER_FILES} ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${SOURCE_FILES})

# For files generated by the moc
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon SofaBaseVisual SofaLoader SofaMiscForceField)
target_link_libraries(${PROJECT_NAME} PUBLIC ${QT_TARGETS})

if(SOFA_DUMP_VISITOR_INFO)
    target_link_libraries(${PROJECT_NAME} PUBLIC tinyxml)
endif()

if(NOT SOFA_NO_OPENGL)
    if(SOFAGUI_QGLVIEWER)
        target_link_libraries(${PROJECT_NAME} PUBLIC QGLViewer)
    endif()
endif()

if(SOFAGUIQT_ENABLE_NODEGRAPH)
    target_link_libraries(${PROJECT_NAME} PUBLIC NodeEditor::nodes)
    if (SOFA_BUILD_RELEASE_PACKAGE OR CMAKE_SYSTEM_NAME STREQUAL Windows)
        sofa_install_libraries(TARGETS NodeEditor::nodes)
    endif()
endif()


# FFMPEG
if(FFMPEG_EXEC_FOUND)
    install(PROGRAMS "${FFMPEG_EXEC_FILE}" DESTINATION bin COMPONENT applications)
endif()

# Create build and install versions of .ini file for resources finding
set(RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${SRC_ROOT}/resources")
set(FFMPEG_EXEC_PATH "${FFMPEG_EXEC_FILE}") # absolute path for build dir, see .ini file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/${PROJECT_NAME}.ini")
set(RESOURCES_DIR "../share/sofa/gui/qt")
get_filename_component(FFMPEG_EXEC_FILENAME "${FFMPEG_EXEC_FILE}" NAME)
set(FFMPEG_EXEC_PATH "../bin/${FFMPEG_EXEC_FILENAME}") # relative path for install dir, see .ini file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini" DESTINATION etc RENAME ${PROJECT_NAME}.ini COMPONENT applications)

install(DIRECTORY "${SRC_ROOT}/resources/" DESTINATION "share/sofa/gui/qt" COMPONENT resources)

sofa_add_targets_to_package(
    PACKAGE_NAME SofaGui
    TARGETS ${PROJECT_NAME} AUTO_SET_TARGET_PROPERTIES
    INCLUDE_SOURCE_DIR "src"
    INCLUDE_INSTALL_DIR "SofaGui"
    )

# Qt deployment
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    include(windeployqt)
    windeployqt(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/bin bin)
endif()
