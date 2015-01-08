#include <QtWidgets/QApplication>
#include <Application.h>

int main(int argc, char **argv)
{
	QApplication app(argc, argv);
	app.setOrganizationName("Sofa");
	app.setApplicationName("qtQuickSofa");
	
    Application sofaApplication(QUrl("qrc:/qml/Main.qml"));

    return app.exec();
}
