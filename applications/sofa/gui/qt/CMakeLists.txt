cmake_minimum_required(VERSION 3.1)
project(SofaGuiQt)

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
    QSofaRecorder.h
    QSofaStatWidget.h
    QTabulationModifyObject.h
    QTransformationWidget.h
    RealGUI.h
    SimpleDataWidget.h
    SofaMouseManager.h    
    SofaVideoRecorderManager.h
    SofaPluginManager.h
    WDoubleLineEdit.h
    panels/QDocBrowser.h
    SofaWindowProfiler.h
)

# these header files do not need MOCcing
set(HEADER_FILES
    FileManagement.h
    GraphListenerQListView.h
    PickHandlerCallBacks.h
    QMenuFilesRecentlyOpened.h
    QModelViewTableDataContainer.h
    SofaGUIQt.h
    StructDataWidget.h
    TableDataWidget.h
    viewer/SofaViewer.h
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
    QSofaRecorder.cpp
    QSofaStatWidget.cpp
    QTabulationModifyObject.cpp
    QTransformationWidget.cpp
    RealGUI.cpp
    SimpleDataWidget.cpp
    SofaMouseManager.cpp
    SofaPluginManager.cpp
    SofaVideoRecorderManager.cpp
    SofaWindowProfiler.cpp
    StructDataWidget.cpp
    TableDataWidget.cpp
    WDoubleLineEdit.cpp
    viewer/SofaViewer.cpp
    panels/QDocBrowser.cpp
)

set(UI_FILES
    BaseGenGraphForm.ui
    DialogAddObject.ui
    GUI.ui
    MouseManager.ui
    PluginManager.ui
    VideoRecorderManager.ui
    WindowProfiler.ui
)


if(SOFA_DUMP_VISITOR_INFO)
    list(APPEND MOC_HEADER_FILES

         WindowVisitor.h

         QVisitorControlPanel.h
         PieWidget.h)

    list(APPEND HEADER_FILES

         GraphVisitor.h)

    list(APPEND SOURCE_FILES

         GraphVisitor.cpp
         WindowVisitor.cpp
         QVisitorControlPanel.cpp
         PieWidget.cpp)

    list(APPEND UI_FILES VisitorGUI.ui)
endif()


if(SOFAGUI_ENABLE_QWT)

    list(APPEND MOC_HEADER_FILES
         GraphDataWidget.h
         QGraphStatWidget.h
         QEnergyStatWidget.h
         QMomentumStatWidget.h)

    list(APPEND SOURCE_FILES
         GraphDataWidget.cpp
         QGraphStatWidget.cpp
         QEnergyStatWidget.cpp
         QMomentumStatWidget.cpp
         )
endif()

# QtViewer
if(SOFAGUI_QTVIEWER)
    list(APPEND MOC_HEADER_FILES viewer/qt/QtViewer.h)
    list(APPEND SOURCE_FILES viewer/qt/QtViewer.cpp)
endif()

# QGLViewer
if(SOFAGUI_QGLVIEWER)
    find_package(QGLViewer)

    list(APPEND MOC_HEADER_FILES viewer/qgl/QtGLViewer.h)
    list(APPEND SOURCE_FILES viewer/qgl/QtGLViewer.cpp)
endif()

if(SOFA_DUMP_VISITOR_INFO)
endif()

set(QT_USE_IMPORTED_TARGETS 1)

find_package(FFMPEGexec QUIET)

find_package(Qt5 COMPONENTS Core QUIET)

if(Qt5Core_FOUND)
    message("SofaGUIQt: Using Qt5")
    set(QT5_NO_LINK_QTMAIN 1)
    find_package(Qt5 COMPONENTS Gui OpenGL Charts REQUIRED)

    set(QT_TARGETS Qt5::Core Qt5::Gui Qt5::OpenGL Qt5::Charts)
    set(EXT_QT_INCLUDES ${Qt5Core_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS} ${Qt5Charts_INCLUDE_DIRS})
    set(EXT_QT_LIBRARIES ${Qt5Core_LIBRARIES} ${Qt5Widgets_LIBRARIES} ${Qt5OpenGL_LIBRARIES} ${Qt5Charts_LIBRARIES})
    # target_include_directories(${EXT_QT_INCLUDES})

    qt5_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    set(MOC_FILES_GROUP "Generated")
    qt5_wrap_ui(FORM_FILES ${UI_FILES})
    set(FORM_FILES_GROUP "Generated")

    if(WIN32)
        sofa_copy_libraries_from_targets(${QT_TARGETS})
    endif()
    sofa_install_libraries_from_targets(${QT_TARGETS})

    # QT INSTALL (dirty)
    # Get Qt libraries dir
    get_target_property(QtCore_location Qt5::Core LOCATION)
    get_filename_component(QT_LIBRARIES_DIR ${QtCore_location} DIRECTORY)
    # Get Qt plugins dir
    list(GET Qt5Gui_PLUGINS 1 QT_PLUGIN_ANY)
    if(TARGET ${QT_PLUGIN_ANY})
        get_target_property(QT_PLUGIN_ANY_LOCATION ${QT_PLUGIN_ANY} LOCATION)
        get_filename_component(QT_PLUGINS_SUBDIR ${QT_PLUGIN_ANY_LOCATION} DIRECTORY)
        get_filename_component(QT_PLUGINS_DIR ${QT_PLUGINS_SUBDIR} DIRECTORY)
    else()
        # Try to guess QT_PLUGINS_DIR
        if(UNIX AND NOT APPLE AND EXISTS "${QT_LIBRARIES_DIR}/qt5/plugins")
            set(QT_PLUGINS_DIR "${QT_LIBRARIES_DIR}/qt5/plugins")
        endif()
    endif()
    # Install Qt plugins
    install(DIRECTORY "${QT_PLUGINS_DIR}/" DESTINATION "bin" COMPONENT applications)
    # Install Qt XCB libraries
    file(GLOB QT_XCB_LIBRARIES
        "${QT_LIBRARIES_DIR}/libQt5DBus${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${QT_LIBRARIES_DIR}/libQt5XcbQpa${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )
    sofa_install_libraries(${QT_XCB_LIBRARIES})
    # Install Qt ICU libraries
    file(GLOB QT_ICU_LIBRARIES
        "${QT_LIBRARIES_DIR}/libicudata${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${QT_LIBRARIES_DIR}/libicui18n${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${QT_LIBRARIES_DIR}/libicuuc${CMAKE_SHARED_LIBRARY_SUFFIX}"
        )
    sofa_install_libraries(${QT_ICU_LIBRARIES})
endif()

add_library(${PROJECT_NAME} SHARED ${MOC_HEADER_FILES} ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${SOURCE_FILES})
# For files generated by the moc
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")
target_include_directories(${PROJECT_NAME} PUBLIC "$<INSTALL_INTERFACE:include>")
target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon SofaBaseVisual SofaLoader SofaOpenglVisual SofaMiscForceField)
target_link_libraries(${PROJECT_NAME} PUBLIC ${EXT_QT_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PRIVATE SofaExporter)
if(SOFAGUI_ENABLE_QWT)
    target_link_libraries(${PROJECT_NAME} PUBLIC qwt)
endif()
if(SOFAGUI_QGLVIEWER)
    target_link_libraries(${PROJECT_NAME} PUBLIC QGLViewer)
endif()

# target_compile_definitions(${PROJECT_NAME} PUBLIC "SOFA_QT4")
# target_compile_definitions(${PROJECT_NAME} PUBLIC "QT3_SUPPORT")

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIQT")

sofa_install_targets(SofaGui SofaGuiQt "sofa/gui/qt")

# Create build and install versions of .ini file for resources finding
set(RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/${PROJECT_NAME}.ini")
if(APPLE AND SOFA_BUILD_RELEASE_PACKAGE)
    set(RESOURCES_DIR "share/sofa/gui/qt")
else()
    set(RESOURCES_DIR "../share/sofa/gui/qt")
endif()
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini" DESTINATION etc RENAME ${PROJECT_NAME}.ini COMPONENT applications)

install(DIRECTORY "resources/" DESTINATION "share/sofa/gui/qt" COMPONENT resources)
install(FILES ${HEADER_FILES} DESTINATION "include/sofa/gui/qt" COMPONENT headers)
install(FILES ${MOC_HEADER_FILES} DESTINATION "include/sofa/gui/qt" COMPONENT headers)
