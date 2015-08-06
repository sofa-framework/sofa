# cmake_minimum_required(VERSION 3.0.2)

include(${SOFA_CMAKE_DIR}/preProject.cmake)

set(HEADER_FILES

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
	WDoubleLineEdit.h
	SofaPluginManager.h
	SofaMouseManager.h
	SofaVideoRecorderManager.h
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
	QModelViewTableUpdater.h

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
	QEnergyStatWidget.cpp              
	QMomentumStatWidget.cpp
	QGraphStatWidget.cpp
	QModelViewTableUpdater.cpp
	)
	
set(UI_FILES

	GUI.ui
	BaseGenGraphForm.ui
	DialogAddObject.ui
	PluginManager.ui
	MouseManager.ui
	VideoRecorderManager.ui
	)

if(SOFA-MISC_DUMP_VISITOR_INFO)
	list(APPEND HEADER_FILES

		 WindowVisitor.h
		 QVisitorControlPanel.h
		 PieWidget.h
		 )
		
	list(APPEND SOURCE_FILES
	
		 GraphVisitor.cpp
		 WindowVisitor.cpp
		 QVisitorControlPanel.cpp
		 PieWidget.cpp
		 )
	
	list(APPEND UI_FILES VisitorGUI.ui)
endif()


if(SOFA-LIB_GUI_QTVIEWER)
	list(APPEND HEADER_FILES viewer/qt/QtViewer.h)
# 	
	list(APPEND SOURCE_FILES viewer/qt/QtViewer.cpp)
	
	AddCompilerDefinitions("SOFA_GUI_QTVIEWER")
endif()

if(SOFA-LIB_GUI_QGLVIEWER)
	list(APPEND HEADER_FILES viewer/qgl/QtGLViewer.h)

	list(APPEND SOURCE_FILES viewer/qgl/QtGLViewer.cpp)
	
	AddCompilerDefinitions("SOFA_GUI_QGLVIEWER")
	AddLinkerDependencies(QGLViewer)
endif()

#UseQt5()

UseQt()

SOFA_QT_WRAP_CPP(MOC_FILES ${HEADER_FILES})
SOFA_QT_WRAP_UI(FORM_FILES ${UI_FILES})

#SOFA_QT4_WRAP_UI(FORM_FILES ${UI_FILES})

# make mocced header / form files path from absolute to relative
ToRelativePath(MOC_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${MOC_FILES}")
ToRelativePath(FORM_FILES "${CMAKE_CURRENT_SOURCE_DIR}" "${FORM_FILES}")

# these header files do not need MOCcing
list(APPEND HEADER_FILES

	viewer/VisualModelPolicy.h
	viewer/SofaViewer.h
	GraphListenerQListView.h
	SofaGUIQt.h
	StructDataWidget.h
	TableDataWidget.h
	FileManagement.h
	PickHandlerCallBacks.h
	QMenuFilesRecentlyOpened.h
	ImageQt.h
	QModelViewTableDataContainer.h
	)

if(SOFA-MISC_DUMP_VISITOR_INFO)
    list(APPEND HEADER_FILES  GraphVisitor.h  )
endif()

#set(MOC_FILES_GROUP "Generated")
#set(FORM_FILES_GROUP "Generated")

add_library(${PROJECT_NAME} SHARED  ${HEADER_FILES} ${MOC_FILES} ${FORM_FILES} ${SOURCE_FILES})

AddCompilerDefinitions("SOFA_BUILD_SOFAGUIQT")
AddLinkerDependencies(SofaGuiCommon Qwt SofaBaseVisual SofaExporter SofaLoader SofaOpenglVisual SofaMiscForceField)


include(${SOFA_CMAKE_DIR}/postProject.cmake)

