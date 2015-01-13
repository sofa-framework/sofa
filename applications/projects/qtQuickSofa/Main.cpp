#include <QtWidgets/QApplication>
#include <QSurfaceFormat>
#include <Application.h>
#include <QProcessEnvironment>
#include <QDebug>

int main(int argc, char **argv)
{
    // TODO: this command disable the multithreaded render loop, currently we need this (especially on Linux/OSX) because our implementation of the sofa interface is not thread-safe
    qputenv("QML_BAD_GUI_RENDER_LOOP", "1");

	QApplication app(argc, argv);
	app.setOrganizationName("Sofa");
	app.setApplicationName("qtQuickSofa");
	
    Application sofaApplication(QUrl("qrc:/qml/Main.qml"));

    return app.exec();
}
