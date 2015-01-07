#include "Viewer.h"
#include "Scene.h"
#include "PythonInteractor.h"

#include <QtWidgets/QApplication>
#include <QQmlApplicationEngine>
#include <QQuickWindow>
#include <QSettings>
#include <QDir>
#include <QFile>
//#include <QOpenGLContext>

int main(int argc, char **argv)
{
	QString organizationName = "Sofa";
	QString applicationName = "qtQuickSofa";
	QString configDir = "./user/config/";

    QApplication app(argc, argv);
	app.setOrganizationName(organizationName);
	app.setApplicationName(applicationName);
	QSettings::setPath(QSettings::Format::IniFormat, QSettings::Scope::UserScope, configDir);
	QSettings::setDefaultFormat(QSettings::Format::IniFormat);
	
	QDir dir;
	QString configFilePath = dir.absolutePath() + "/" + configDir + organizationName + "/" + applicationName + ".ini";
	
	QFileInfo fileInfo(configFilePath);
	if(!fileInfo.isFile())
	{
		QFile file(":/data/config/default.ini");
		if(!file.open(QFile::OpenModeFlag::ReadOnly))
		{
			qDebug() << "ERROR: the default config file has not been found!";
		}
		else
		{
			dir.mkpath(configDir + organizationName);
			if(!file.copy(configFilePath))
				qDebug() << "ERROR: the config file could not be created!";
			else
				if(!QFile::setPermissions(configFilePath, QFile::ReadOwner | QFile::WriteOwner | QFile::ReadUser | QFile::WriteUser | QFile::ReadGroup | QFile::WriteGroup))
					qDebug() << "ERROR: cannot set permission on the config file!";
		}
	}
	
    QUrl mainScriptUrl = QUrl("qrc:///data/qml/Main.qml");

	/*QSurfaceFormat format;
	format.setMajorVersion(3);
	format.setMajorVersion(2);
	format.setProfile(QSurfaceFormat::OpenGLContextProfile::CompatibilityProfile);*/

    QQmlApplicationEngine engine;
	engine.addImportPath("qrc:///data/qml/component/");
	engine.load(mainScriptUrl);

    QObject* topLevel = engine.rootObjects().value(0);
    QQuickWindow* window = qobject_cast<QQuickWindow *>(topLevel);
    if(0 == window)
    {
        qDebug() << "Your QML root object should be a window";
        return 1;
    }

	//window->setFormat(format);
    window->show();

    return app.exec();
}
