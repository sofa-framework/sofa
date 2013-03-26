cmake_minimum_required(VERSION 2.8)

project("SofaGUIQt")

include(${SOFA_CMAKE_DIR}/pre.cmake)

set(HEADER_FILES

	viewer/VisualModelPolicy.h
	viewer/SofaViewer.h
	GraphListenerQListView.h
	GenGraphForm.h
	AddObject.h
	RealGUI.h
	LinkWidget.h
	DataWidget.h
	DataFilenameWidget.h
	DisplayFlagsDataWidget.h
	GraphDataWidget.h
	MaterialDataWidget.h
	ModifyObject.h
	SimpleDataWidget.h
	SofaGUIQt.h
	StructDataWidget.h
	TableDataWidget.h
	WDoubleLineEdit.h 
	FileManagement.h
	SofaPluginManager.h
	SofaMouseManager.h
	SofaVideoRecorderManager.h	
	PickHandlerCallBacks.h
	QDataDescriptionWidget.h
	QDisplayPropertyWidget.h
	QDisplayLinkWidget.h     
	QDisplayDataWidget.h
	QGraphStatWidget.h
	QEnergyStatWidget.h              
	QMomentumStatWidget.h 
	QTabulationModifyObject.h
	QTransformationWidget.h
	QMouseOperations.h
	QSofaListView.h
	QSofaRecorder.h
	QSofaStatWidget.h
	QMenuFilesRecentlyOpened.h
	ImageQt.h 
	)

set(SOURCE_FILES

	viewer/SofaViewer.cpp
	GraphListenerQListView.cpp
	GenGraphForm.cpp
	AddObject.cpp
	RealGUI.cpp
	LinkWidget.cpp 
	DataWidget.cpp 
	DataFilenameWidget.cpp
	DisplayFlagsDataWidget.cpp
	GraphDataWidget.cpp  
	MaterialDataWidget.cpp
	ModifyObject.cpp
	SimpleDataWidget.cpp
	StructDataWidget.cpp
	TableDataWidget.cpp
	WDoubleLineEdit.cpp
	FileManagement.cpp
	SofaPluginManager.cpp
	SofaVideoRecorderManager.cpp
	SofaMouseManager.cpp
	QDataDescriptionWidget.cpp
	PickHandlerCallBacks.cpp
	QDisplayPropertyWidget.cpp
	QDisplayLinkWidget.cpp
	QDisplayDataWidget.cpp
	QMouseOperations.cpp               
	QTabulationModifyObject.cpp
	QTransformationWidget.cpp
	QSofaListView.cpp
	QSofaRecorder.cpp
	QSofaStatWidget.cpp
	QMenuFilesRecentlyOpened.cpp
	ImageQt.cpp 
	)
	
set(FORM_FILES

	GUI.ui
	BaseGenGraphForm.ui
	DialogAddObject.ui
	PluginManager.ui
	MouseManager.ui
	VideoRecorderManager.ui
	)

if(SOFA_DUMP_VISITOR_INFO)
	list(APPEND FORM_FILES VisitorGUI.ui)
endif()

set(SOFA_QT4 1) # TODO: remove this line
if(SOFA_QT4)
	find_package(Qt4 PATHS "C:/QtLibx64SP1" REQUIRED)
	
	QT4_WRAP_CPP(MOC_HEADER_FILES ${HEADER_FILES})
	QT4_WRAP_UI(FORM_HEADER_FILES ${FORM_FILES})
	
	#CONFIG += qt uic uic3
	SET(QT_USE_QTOPENGL TRUE)
	SET(QT_USE_QT3SUPPORT TRUE)
	SET(QT_USE_QTXML TRUE)
	#QT += opengl qt3support xml
else()
	find_package(Qt3 PATHS "C:/QtLibx64SP1" REQUIRED)
	
	QT3_WRAP_CPP(MOC_HEADER_FILES ${HEADER_FILES})
	QT3_WRAP_UI(FORM_HEADER_FILES ${FORM_FILES})

	#CONFIG += qt
	SET(QT_USE_QTOPENGL TRUE)
endif()

include(${QT_USE_FILE})
add_definitions(${QT_DEFINITIONS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(SOFA_GUI_QTVIEWER)
	list(APPEND HEADER_FILES viewer/qt/QtViewer.h)
	
	list(APPEND SOURCE_FILES viewer/qt/QtViewer.cpp)
endif()

if(SOFA_GUI_QGLVIEWER)
	list(APPEND HEADER_FILES viewer/qgl/QtGLViewer.h)
	
	list(APPEND SOURCE_FILES viewer/qgl/QtGLViewer.cpp)
endif()

add_library(${PROJECT_NAME} SHARED ${MOC_HEADER_FILES} ${SOURCE_FILES})

set(COMPILER_DEFINES "SOFA_BUILD_SOFAGUIQT")
set(LINKER_DEPENDENCIES SofaGUICommon SofaGUIQt SofaGUIGlut ${QT_LIBRARIES})

include(${SOFA_CMAKE_DIR}/post.cmake)

