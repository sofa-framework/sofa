#include "Application.h"
#include "SofaQtQuickGUI.h"

#include <qqml.h>
#include <QDebug>
#include <QQmlContext>
#include <QQmlEngine>
#include <QSettings>
#include <QDir>
#include <QFile>
#include <QQuickWindow>

Q_IMPORT_PLUGIN(SofaQtQuickGUI);

Application::Application(QObject* parent) : QQmlApplicationEngine(parent)
{
	init();
}

Application::Application(const QUrl& url, QObject* parent) : QQmlApplicationEngine(parent)
{
	init();
	load(url);
}

Application::Application(const QString& filePath, QObject* parent) : QQmlApplicationEngine(parent)
{
	init();
	load(filePath);
}

void Application::init()
{
	if(!qApp)
	{
		qFatal("Application must be instantiated AFTER a QApplication");
		return;
	}

	qt_static_plugin_SofaQtQuickGUI().instance();

	QString organizationName = qApp->organizationName();
	QString applicationName = qApp->applicationName();
	QString configDir = "./user/config/";

	QSettings::setPath(QSettings::Format::IniFormat, QSettings::Scope::UserScope, configDir);
	QSettings::setDefaultFormat(QSettings::Format::IniFormat);

	QDir dir;
	QString configFilePath = dir.absolutePath() + "/" + configDir + organizationName + "/" + applicationName + ".ini";

	QFileInfo fileInfo(configFilePath);
	if(!fileInfo.isFile())
	{
		QFile file(":/config/default.ini");
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

	addImportPath("qrc:/");

	rootContext()->setContextProperty("application", this);
}

Application::~Application()
{
	clearSettingGroup("dummy");
}

void Application::trimCache(QObject* object)
{
	if(!object)
		object = this;

	QQmlContext* context = QQmlEngine::contextForObject(object);
	if(!context)
		return;

	QQmlEngine* engine = context->engine();
	if(!engine)
		return;

	engine->trimComponentCache();
}

void Application::clearSettingGroup(const QString& group)
{
	QSettings settings;
	settings.beginGroup(group);
	settings.remove("");
	settings.endGroup();
}