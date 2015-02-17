#include <QtCore/QCoreApplication>
#include <QtWidgets/QApplication>
#include <QQmlApplicationEngine>
#include <QPluginLoader>
#include <QDebug>
#include "Tools.h"

using namespace sofa::qtquick;

int main(int argc, char **argv)
{
    // TODO: this command disable the multithreaded render loop, currently we need this on Linux/OSX because our implementation of the sofa interface is not thread-safe
    qputenv("QML_BAD_GUI_RENDER_LOOP", "1");

	QApplication app(argc, argv);
    app.addLibraryPath(QCoreApplication::applicationDirPath() + "/../lib/");

    // application specific settings
	app.setOrganizationName("Sofa");
	app.setApplicationName("qtQuickSofa");

    QSettings::setPath(QSettings::Format::IniFormat, QSettings::Scope::UserScope, QCoreApplication::applicationDirPath() + "/config/");
    QSettings::setDefaultFormat(QSettings::Format::IniFormat);

    // use the default.ini settings if it is the first time the user launch the application
    Tools::useDefaultSettingsAtFirstLaunch();

    // plugin initialization
	QString pluginName("SofaQtQuickGUI");
#ifdef SOFA_LIBSUFFIX
	pluginName += sofa_tostring(SOFA_LIBSUFFIX);
#endif
	QPluginLoader pluginLoader(pluginName);

    // first call to instance() initialize the plugin
    if(0 == pluginLoader.instance()) {
        qCritical() << "SofaQtQuickGUI plugin has not been found!";
        return -1;
    }

    // launch the main script
    QQmlApplicationEngine applicationEngine;
    applicationEngine.addImportPath("qrc:/");
    applicationEngine.addImportPath(QCoreApplication::applicationDirPath() + "/../lib/qml/");
    applicationEngine.load(QUrl("qrc:/qml/Main.qml"));

    return app.exec();
}
