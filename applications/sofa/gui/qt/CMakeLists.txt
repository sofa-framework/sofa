cmake_minimum_required(VERSION 3.1)
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
            add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/libQGLViewer-2.7.1/QGLViewer ${CMAKE_BINARY_DIR}/extlibs/QGLViewer)
        endif()
    endif()
else()
    set(SOFAGUI_QTVIEWER OFF)
    set(SOFAGUI_QGLVIEWER OFF)
endif()
sofa_set_01(SOFAGUIQT_HAVE_QTVIEWER VALUE ${SOFAGUI_QTVIEWER} BOTH_SCOPES)
sofa_set_01(SOFAGUIQT_HAVE_QGLVIEWER VALUE ${SOFAGUI_QGLVIEWER} BOTH_SCOPES)

# QWT
# TODO: rename SOFAGUI_ENABLE_QWT into SOFAGUIQT_ENABLE_QWT
option(SOFAGUI_ENABLE_QWT "Enable QWT for Qt GUIs")
sofa_set_01(SOFAGUIQT_HAVE_QWT VALUE ${SOFAGUI_ENABLE_QWT} BOTH_SCOPES) # see SofaGuiQt.h.in
if(SOFAGUI_ENABLE_QWT)
    message("Qwt activated")
    find_package(Qwt QUIET)
    if(NOT QWT_FOUND)
        add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/qwt-6.1.4/src ${CMAKE_BINARY_DIR}/extlibs/qwt)
    endif()
endif()

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



#NodeEditor
option(SOFAGUIQT_ENABLE_NODEGRAPH "Build the Node graph representation. NodeEditor library is needed." OFF)
if (SOFAGUIQT_ENABLE_NODEGRAPH)
    find_package(NodeEditor REQUIRED)
endif()
sofa_set_01(SOFAGUIQT_HAVE_NODEEDITOR VALUE ${SOFAGUIQT_ENABLE_NODEGRAPH} BOTH_SCOPES)

# FFMPEG_exec
sofa_find_package(FFMPEG_exec BOTH_SCOPES)

set(MOC_HEADER_FILES
    AddObject.h
    DataFilenameWidget.h
    DataWidget.h
    DisplayFlagsDataWidget.h
    GenGraphForm.h
    LinkWidget.h
    MaterialDataWidget.h
    RGBAColorDataWidget.h
    ModifyObject.h
    QDataDescriptionWidget.h
    QDisplayDataWidget.h
    QDisplayLinkWidget.h
    QDisplayPropertyWidget.h
    QRGBAColorPicker.h
    QModelViewTableUpdater.h
    QMouseOperations.h
    QMouseWheelAdjustementGuard.h
    QSofaListView.h
    QSofaStatWidget.h
    QTabulationModifyObject.h
    QTransformationWidget.h
    RealGUI.h
    SimpleDataWidget.h
    SofaMouseManager.h    
    SofaVideoRecorderManager.h
    SofaPluginManager.h
    WDoubleLineEdit.h    
    )
set(HEADER_FILES
    SofaGuiQt.h.in # = config.h.in
    FileManagement.h
    GraphListenerQListView.h
    PickHandlerCallBacks.h
    QMenuFilesRecentlyOpened.h
    QModelViewTableDataContainer.h
    StructDataWidget.h
    TableDataWidget.h
    viewer/SofaViewer.h
    viewer/EngineBackend.h
    viewer/VisualModelPolicy.h
    GenericWidget.h
    )
set(SOURCE_FILES
    AddObject.cpp
    DataFilenameWidget.cpp
    DataWidget.cpp
    DisplayFlagsDataWidget.cpp
    FileManagement.cpp
    GenGraphForm.cpp
    GraphListenerQListView.cpp
    LinkWidget.cpp
    MaterialDataWidget.cpp
    RGBAColorDataWidget.cpp
    ModifyObject.cpp
    PickHandlerCallBacks.cpp
    QDataDescriptionWidget.cpp
    QDisplayDataWidget.cpp
    QDisplayLinkWidget.cpp
    QDisplayPropertyWidget.cpp
    QRGBAColorPicker.cpp
    QMenuFilesRecentlyOpened.cpp
    QModelViewTableUpdater.cpp
    QMouseOperations.cpp
    QMouseWheelAdjustementGuard.cpp
    QSofaListView.cpp
    QSofaStatWidget.cpp
    QTabulationModifyObject.cpp
    QTransformationWidget.cpp
    RealGUI.cpp
    SimpleDataWidget.cpp
    SofaMouseManager.cpp
    SofaPluginManager.cpp
    SofaVideoRecorderManager.cpp    
    StructDataWidget.cpp
    TableDataWidget.cpp
    WDoubleLineEdit.cpp
    viewer/SofaViewer.cpp
    viewer/VisualModelPolicy.cpp    
)
set(UI_FILES
    BaseGenGraphForm.ui
    DialogAddObject.ui
    GUI.ui
    MouseManager.ui
    PluginManager.ui
    VideoRecorderManager.ui    
)

if(SOFA_DUMP_VISITOR_INFO)
    list(APPEND MOC_HEADER_FILES
         WindowVisitor.h
         QVisitorControlPanel.h
         PieWidget.h
         )
    list(APPEND HEADER_FILES
         GraphVisitor.h
         )
    list(APPEND SOURCE_FILES
         GraphVisitor.cpp
         WindowVisitor.cpp
         QVisitorControlPanel.cpp
         PieWidget.cpp
         )
    list(APPEND UI_FILES
        VisitorGUI.ui
        )
endif()

if(NOT SOFA_NO_OPENGL)
    list(APPEND HEADER_FILES
        GLPickHandler.h
        viewer/GLBackend.h
        viewer/OglModelPolicy.h
        )
    list(APPEND SOURCE_FILES
        GLPickHandler.cpp
        viewer/GLBackend.cpp
        viewer/OglModelPolicy.cpp
        )

    # QtViewer
    if(SOFAGUI_QTVIEWER)
        list(APPEND MOC_HEADER_FILES viewer/qt/QtViewer.h)
        list(APPEND SOURCE_FILES viewer/qt/QtViewer.cpp)
    endif()

    # QGLViewer
    if(SOFAGUI_QGLVIEWER)
        list(APPEND MOC_HEADER_FILES viewer/qgl/QtGLViewer.h)
        list(APPEND SOURCE_FILES viewer/qgl/QtGLViewer.cpp)
    endif()
endif()

# QWT
if(SOFAGUI_ENABLE_QWT)
    list(APPEND MOC_HEADER_FILES
         GraphDataWidget.h
         QGraphStatWidget.h
         QEnergyStatWidget.h
         QMomentumStatWidget.h
         )
    list(APPEND SOURCE_FILES
         GraphDataWidget.cpp
         QGraphStatWidget.cpp
         QEnergyStatWidget.cpp
         QMomentumStatWidget.cpp
         )
endif()

# QtCharts
if(Qt5Charts_FOUND)
    list(APPEND MOC_HEADER_FILES SofaWindowProfiler.h)
    list(APPEND SOURCE_FILES SofaWindowProfiler.cpp)
    list(APPEND UI_FILES WindowProfiler.ui)
else()
    message(STATUS "SofaGuiQt: QtCharts not found, SofaWindowProfiler will not be built.")
endif()

if(SOFAGUIQT_ENABLE_QDOCBROWSER)
    list(APPEND MOC_HEADER_FILES panels/QDocBrowser.h)
    list(APPEND SOURCE_FILES panels/QDocBrowser.cpp)
else()
    message(STATUS "SofaGuiQt: Qt5WebEngine not found, QDocBrowser will not be built.")
endif()


#NodeEditor
if (SOFAGUIQT_ENABLE_NODEGRAPH)
    list(APPEND MOC_HEADER_FILES 
        SofaWindowDataGraph.h
        dataGraph/SofaComponentNodeModel.h
        )

    list(APPEND SOURCE_FILES 
        SofaWindowDataGraph.cpp
        dataGraph/SofaComponentNodeModel.cpp
        )
endif()

qt5_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
set(MOC_FILES_GROUP "Generated")
qt5_wrap_ui(FORM_FILES ${UI_FILES})
set(FORM_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} SHARED ${MOC_HEADER_FILES} ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${SOURCE_FILES})

# For files generated by the moc
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon SofaBaseVisual SofaLoader SofaMiscForceField)
target_link_libraries(${PROJECT_NAME} PUBLIC ${QT_TARGETS})

if(SOFAGUI_ENABLE_QWT)
    target_link_libraries(${PROJECT_NAME} PUBLIC qwt)
endif()
if(NOT SOFA_NO_OPENGL)
    if(SOFAGUI_QGLVIEWER)
        target_link_libraries(${PROJECT_NAME} PUBLIC QGLViewer)
    endif()
endif()

if(NODEEDITOR_FOUND)
    target_link_libraries(${PROJECT_NAME} PUBLIC ${NODEEDITOR_LIBRARY})
    sofa_install_libraries(PATHS ${NODEEDITOR_LIBRARY})
    target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${NODEEDITOR_INCLUDE_PATH}>")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIQT")

# FFMPEG
if(FFMPEG_EXEC_FOUND)
    install(PROGRAMS "${FFMPEG_EXEC_FILE}" DESTINATION bin COMPONENT applications)
endif()

# Create build and install versions of .ini file for resources finding
set(RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")
set(FFMPEG_EXEC_PATH "${FFMPEG_EXEC_FILE}") # absolute path for build dir, see .ini file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/${PROJECT_NAME}.ini")
set(RESOURCES_DIR "../share/sofa/gui/qt")
get_filename_component(FFMPEG_EXEC_FILENAME "${FFMPEG_EXEC_FILE}" NAME)
set(FFMPEG_EXEC_PATH "../bin/${FFMPEG_EXEC_FILENAME}") # relative path for install dir, see .ini file
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini" DESTINATION etc RENAME ${PROJECT_NAME}.ini COMPONENT applications)

install(DIRECTORY "resources/" DESTINATION "share/sofa/gui/qt" COMPONENT resources)

sofa_install_targets(SofaGui SofaGuiQt "SofaGui/sofa/gui/qt")

# Qt deployment
if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    include(windeployqt)
    windeployqt(${PROJECT_NAME} ${CMAKE_BINARY_DIR}/bin ${CMAKE_INSTALL_PREFIX}/bin)
endif()
