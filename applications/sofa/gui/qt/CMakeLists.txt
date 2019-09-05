cmake_minimum_required(VERSION 3.1)
project(SofaGuiQt)

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
sofa_set_bool(SOFAGUIQT_HAVE_QTVIEWER ${SOFAGUI_QTVIEWER} BOTH_SCOPES)
sofa_set_bool(SOFAGUIQT_HAVE_QGLVIEWER ${SOFAGUI_QGLVIEWER} BOTH_SCOPES)

# QWT
# TODO: rename SOFAGUI_ENABLE_QWT into SOFAGUIQT_ENABLE_QWT
option(SOFAGUI_ENABLE_QWT "Enable QWT for Qt GUIs")
sofa_set_bool(SOFAGUIQT_HAVE_QWT ${SOFAGUI_ENABLE_QWT} BOTH_SCOPES) # see SofaGuiQt.h.in
if(SOFAGUI_ENABLE_QWT)
    message("Qwt activated")
    add_subdirectory(${CMAKE_SOURCE_DIR}/extlibs/qwt-6.1.2/src ${CMAKE_BINARY_DIR}/extlibs/qwt)
endif()

# VSync
# TODO: rename SOFAGUI_ENABLE_VSYNC into SOFAGUIQT_ENABLE_VSYNC
option(SOFAGUI_ENABLE_VSYNC "Enable vertical refresh sync" OFF)
sofa_set_bool(SOFAGUIQT_ENABLE_VSYNC_ ${SOFAGUI_ENABLE_VSYNC}) # see SofaGuiQt.h.in

# Qt native menu
# TODO: rename SOFAGUI_ENABLE_NATIVE_MENU into SOFAGUIQT_ENABLE_NATIVE_MENU
set(DEFAULT_SOFAGUI_ENABLE_NATIVE_MENU ON)
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(DEFAULT_SOFAGUI_ENABLE_NATIVE_MENU OFF)
endif()
option(SOFAGUI_ENABLE_NATIVE_MENU "Enable Qt to use the operating system's native menu for the Qt GUI" ${DEFAULT_SOFAGUI_ENABLE_NATIVE_MENU})

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
    panels/QDocBrowser.h
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
    panels/QDocBrowser.cpp
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
sofa_find_package(Qt5 COMPONENTS Charts QUIET BOTH_SCOPES)
if (Qt5Charts_FOUND)
    list(APPEND MOC_HEADER_FILES SofaWindowProfiler.h)
    list(APPEND SOURCE_FILES SofaWindowProfiler.cpp)
    list(APPEND UI_FILES WindowProfiler.ui)

    set(QT_TARGETS ${QT_TARGETS} Qt5::Charts)
    set(EXT_QT_INCLUDES ${EXT_QT_INCLUDES} ${Qt5Charts_INCLUDE_DIRS})
    set(EXT_QT_LIBRARIES ${EXT_QT_LIBRARIES} ${Qt5Charts_LIBRARIES})
else()
    message(STATUS "SofaGuiQt: QtCharts not found, SofaWindowProfiler will not be built.")
endif()

set(QT_USE_IMPORTED_TARGETS 1)
sofa_find_package(Qt5 COMPONENTS Core QUIET)

if(Qt5Core_FOUND)
    message(STATUS "SofaGuiQt: Using Qt5")
    set(QT5_NO_LINK_QTMAIN 1)
    sofa_find_package(Qt5 COMPONENTS Gui OpenGL REQUIRED)

    set(QT_TARGETS ${QT_TARGETS} Qt5::Core Qt5::Gui Qt5::OpenGL)
    set(EXT_QT_INCLUDES ${EXT_QT_INCLUDES} ${Qt5Core_INCLUDE_DIRS} ${Qt5Widgets_INCLUDE_DIRS} ${Qt5OpenGL_INCLUDE_DIRS})
    set(EXT_QT_LIBRARIES ${EXT_QT_LIBRARIES} ${Qt5Core_LIBRARIES} ${Qt5Widgets_LIBRARIES} ${Qt5OpenGL_LIBRARIES} )

    qt5_wrap_cpp(MOC_FILES ${MOC_HEADER_FILES})
    set(MOC_FILES_GROUP "Generated")
    qt5_wrap_ui(FORM_FILES ${UI_FILES})
    set(FORM_FILES_GROUP "Generated")

    # QT INSTALL (dirty)
    sofa_install_libraries(TARGETS ${QT_TARGETS})
    # Get Qt libraries dir
    get_target_property(QtCore_location Qt5::Core LOCATION)
    get_filename_component(QT_LIBRARIES_DIR ${QtCore_location} DIRECTORY)
    get_filename_component(QT_LIBRARIES_DIR_NAME ${QT_LIBRARIES_DIR} NAME)
    if(NOT "${QT_LIBRARIES_DIR_NAME}" STREQUAL "lib")
        get_filename_component(QT_LIBRARIES_DIR ${QT_LIBRARIES_DIR} DIRECTORY) # parent dir
        get_filename_component(QT_LIBRARIES_DIR_NAME ${QT_LIBRARIES_DIR} NAME)
    endif()
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
    if(EXISTS "${QT_PLUGINS_DIR}/iconengines")
        install(DIRECTORY "${QT_PLUGINS_DIR}/iconengines" DESTINATION "bin" COMPONENT applications)
    endif()
    if(EXISTS "${QT_PLUGINS_DIR}/imageformats")
        install(DIRECTORY "${QT_PLUGINS_DIR}/imageformats" DESTINATION "bin" COMPONENT applications)
    endif()
    if(EXISTS "${QT_PLUGINS_DIR}/platforms")
        install(DIRECTORY "${QT_PLUGINS_DIR}/platforms" DESTINATION "bin" COMPONENT applications)
    endif()
    if(EXISTS "${QT_PLUGINS_DIR}/styles")
        install(DIRECTORY "${QT_PLUGINS_DIR}/styles" DESTINATION "bin" COMPONENT applications)
    endif()
    if(EXISTS "${QT_PLUGINS_DIR}/xcbglintegrations")
        install(DIRECTORY "${QT_PLUGINS_DIR}/xcbglintegrations" DESTINATION "bin" COMPONENT applications)
    endif()
    # Install Apple Qt frameworks
    if(APPLE)
        if(EXISTS "${QT_LIBRARIES_DIR}/QtDBus.framework")
            install(DIRECTORY "${QT_LIBRARIES_DIR}/QtDBus.framework" DESTINATION "lib" COMPONENT appications)
        endif()
        if(EXISTS "${QT_LIBRARIES_DIR}/QtPrintSupport.framework")
            install(DIRECTORY "${QT_LIBRARIES_DIR}/QtPrintSupport.framework" DESTINATION "lib" COMPONENT appications)
        endif()
    endif()
endif()

add_library(${PROJECT_NAME} SHARED ${MOC_HEADER_FILES} ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${SOURCE_FILES})

# For files generated by the moc
target_include_directories(${PROJECT_NAME} PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>")

target_link_libraries(${PROJECT_NAME} PUBLIC SofaGuiCommon SofaBaseVisual SofaLoader SofaMiscForceField)
target_link_libraries(${PROJECT_NAME} PUBLIC ${EXT_QT_LIBRARIES})
if(SOFAGUI_ENABLE_QWT)
    target_link_libraries(${PROJECT_NAME} PUBLIC qwt)
endif()
if(NOT SOFA_NO_OPENGL)
    if(SOFAGUI_QGLVIEWER)
        target_link_libraries(${PROJECT_NAME} PUBLIC QGLViewer)
    endif()
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-DSOFA_BUILD_SOFAGUIQT")

sofa_install_targets(SofaGui SofaGuiQt "SofaGui/sofa/gui/qt")

# Create build and install versions of .ini file for resources finding
set(RESOURCES_DIR "${CMAKE_CURRENT_SOURCE_DIR}/resources")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/${PROJECT_NAME}.ini")
set(RESOURCES_DIR "../share/sofa/gui/qt")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/etc/${PROJECT_NAME}.ini.in "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini")
install(FILES "${CMAKE_BINARY_DIR}/etc/installed${PROJECT_NAME}.ini" DESTINATION etc RENAME ${PROJECT_NAME}.ini COMPONENT applications)

install(DIRECTORY "resources/" DESTINATION "share/sofa/gui/qt" COMPONENT resources)
