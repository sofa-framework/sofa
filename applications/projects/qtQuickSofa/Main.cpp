#include <QtWidgets/QApplication>
#include <QQmlApplicationEngine>
#include <QPluginLoader>
#include <QDebug>
#include "Tools.h"

int main(int argc, char **argv)
{
    // TODO: this command disable the multithreaded render loop, currently we need this on Linux/OSX because our implementation of the sofa interface is not thread-safe
    qputenv("QML_BAD_GUI_RENDER_LOOP", "1");

	QApplication app(argc, argv);
    app.addLibraryPath("../lib/");

    // application specific settings
	app.setOrganizationName("Sofa");
	app.setApplicationName("qtQuickSofa");

    QSettings::setPath(QSettings::Format::IniFormat, QSettings::Scope::UserScope, "./user/config/");
    QSettings::setDefaultFormat(QSettings::Format::IniFormat);

    // use the default.ini settings if it is the first time the user launch the application
    Tools::useDefaultSettingsAtFirstLaunch();

    // plugin initialization
    QPluginLoader pluginLoader("SofaQtQuickGUI");

    // first call to instance() initialize the plugin
    if(0 == pluginLoader.instance())
        qCritical() << "SofaQtQuickGUI plugin has not been found!";

    // launch the main script
    QQmlApplicationEngine applicationEngine;
    applicationEngine.addImportPath("qrc:/");
    applicationEngine.load(QUrl("qrc:/qml/Main.qml"));

    return app.exec();
}
