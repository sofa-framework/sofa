CONFIG       += ordered
TEMPLATE      = subdirs
SUBDIRS       = QGLViewer examples

QT_VERSION=$$[QT_VERSION]

equals (QT_MAJOR_VERSION, 5) {
	cache()
}

!equals (QT_MAJOR_VERSION, 6) {
	SUBDIRS = designerPlugin
}
