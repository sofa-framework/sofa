#include <QtCore/QCoreApplication>
#include <QtWidgets/QApplication>
#include <QQmlApplicationEngine>
#include <QPluginLoader>
#include <QDebug>
#include "Tools.h"

#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/Utils.h>
#include <sofa/gui/BaseGUI.h>

using namespace sofa::qtquick;

int main(int argc, char **argv)
{
    // TODO: this command disable the multithreaded render loop, currently we need this on Linux/OSX because our implementation of the sofa interface is not thread-safe
    qputenv("QML_BAD_GUI_RENDER_LOOP", "1");

    QLocale::setDefault(QLocale(QLocale::English, QLocale::UnitedStates));

	QApplication app(argc, argv);
    app.addLibraryPath(QCoreApplication::applicationDirPath() + "/../lib/");

    // Add the plugin directory to PluginRepository
#ifdef WIN32
    const std::string pluginDir = sofa::helper::Utils::getExecutableDirectory();
#else
    const std::string pluginDir = sofa::helper::Utils::getSofaPathPrefix() + "/lib";
#endif
    sofa::helper::system::PluginRepository.addFirstPath(pluginDir);

    // Initialise paths
    sofa::gui::BaseGUI::setConfigDirectoryPath(sofa::helper::Utils::getSofaPathPrefix() + "/config");
    sofa::gui::BaseGUI::setScreenshotDirectoryPath(sofa::helper::Utils::getSofaPathPrefix() + "/screenshots");

    // sofa init
    sofa::helper::system::DataRepository.addFirstPath("./");
    sofa::helper::system::DataRepository.addLastPath(sofa::helper::Utils::getSofaPathPrefix() + "/share/");
    sofa::helper::system::DataRepository.addLastPath(sofa::helper::Utils::getSofaPathPrefix() + "/examples/");

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
