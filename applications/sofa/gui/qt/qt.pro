load(sofa/pre)

TEMPLATE = lib
TARGET = sofaguiqt

DEFINES += SOFA_BUILD_SOFAGUIQT
INCLUDEPATH += $$ROOT_SRC_DIR/applications
DEPENDPATH += $$ROOT_SRC_DIR/applications
INCLUDEPATH += $$SRC_DIR
DEPENDPATH += $$SRC_DIR

contains(DEFINES, SOFA_QT4) {	
	CONFIG += qt uic uic3
	QT += opengl qt3support xml

	FORMS3 += GUI.ui
	FORMS3 += BaseGenGraphForm.ui
	FORMS3 += DialogAddObject.ui
	FORMS3 += PluginManager.ui
	FORMS3 += MouseManager.ui
	FORMS3 += VideoRecorderManager.ui
	contains(DEFINES, SOFA_DUMP_VISITOR_INFO) {
		FORMS3 += VisitorGUI.ui
	}
}
else {
	CONFIG += qt
	QT += opengl

	FORMS += GUI.ui
	FORMS += BaseGenGraphForm.ui
	FORMS += DialogAddObject.ui
	FORMS += PluginManager.ui
	FORMS += MouseManager.ui
	FORMS += VideoRecorderManager.ui
	contains(DEFINES, SOFA_DUMP_VISITOR_INFO) {
		FORMS += VisitorGUI.ui
	}
}


HEADERS += viewer/VisualModelPolicy.h \
	viewer/SofaViewer.h \
	viewer/ViewerFactory.h \
	GraphListenerQListView.h \
	GenGraphForm.h \
	AddObject.h \
	RealGUI.h \
	LinkWidget.h \
	DataWidget.h \
	DataFilenameWidget.h \
	DisplayFlagsDataWidget.h \
	GraphDataWidget.h \
	MaterialDataWidget.h \
	ModifyObject.h \
	SimpleDataWidget.h \
	SofaGUIQt.h \
	StructDataWidget.h \
	TableDataWidget.h \
	WDoubleLineEdit.h \ 
	FileManagement.h \
	SofaPluginManager.h \
	SofaMouseManager.h \
	SofaVideoRecorderManager.h \	
	PickHandlerCallBacks.h \
	QDataDescriptionWidget.h \
	QDisplayLinkWidget.h \     
	QDisplayDataWidget.h \     
	QEnergyStatWidget.h \              
	QTabulationModifyObject.h \
	QTransformationWidget.h \
	QMouseOperations.h \
	QSofaListView.h \
	QSofaRecorder.h \
	QSofaStatWidget.h \
	QMenuFilesRecentlyOpened.h \
	ImageQt.h \ 
	
SOURCES += viewer/SofaViewer.cpp \
	GraphListenerQListView.cpp \
	GenGraphForm.cpp \
	AddObject.cpp \
	RealGUI.cpp \
	LinkWidget.cpp \ 
	DataWidget.cpp \ 
	DataFilenameWidget.cpp \
	DisplayFlagsDataWidget.cpp \
	GraphDataWidget.cpp \  
	MaterialDataWidget.cpp \
	ModifyObject.cpp \
	SimpleDataWidget.cpp \
	StructDataWidget.cpp \
	TableDataWidget.cpp \
	WDoubleLineEdit.cpp \
	FileManagement.cpp \
	SofaPluginManager.cpp \
	SofaVideoRecorderManager.cpp \
	SofaMouseManager.cpp \
	QDataDescriptionWidget.cpp \
	PickHandlerCallBacks.cpp \
	QDisplayLinkWidget.cpp \
	QDisplayDataWidget.cpp \
	QEnergyStatWidget.cpp \       
	QMouseOperations.cpp \               
	QTabulationModifyObject.cpp \
	QTransformationWidget.cpp \
	QSofaListView.cpp \
	QSofaRecorder.cpp \
	QSofaStatWidget.cpp \
	QMenuFilesRecentlyOpened.cpp \
	ImageQt.cpp \ 

contains(DEFINES, SOFA_QT4) {
	HEADERS += QModelViewTableDataContainer.h \
		QModelViewTableUpdater.h
}

!contains(DEFINES, SOFA_QT4) {
	HEADERS += QTableDataContainer.h \
		QTableUpdater.h
}

contains(DEFINES, SOFA_DUMP_VISITOR_INFO) {
	HEADERS += GraphVisitor.h \
		WindowVisitor.h \
		QVisitorControlPanel.h \
		PieWidget.h
	
	SOURCES += GraphVisitor.cpp \
		WindowVisitor.cpp \
		QVisitorControlPanel.cpp \
		PieWidget.cpp
}

contains(DEFINES, SOFA_GUI_QTVIEWER) {
	SOURCES += viewer/qt/QtViewer.cpp
	HEADERS += viewer/qt/QtViewer.h
}


contains(DEFINES, SOFA_GUI_QGLVIEWER) {
	SOURCES += viewer/qgl/QtGLViewer.cpp
	HEADERS += viewer/qgl/QtGLViewer.h
}

load(sofa/post)
