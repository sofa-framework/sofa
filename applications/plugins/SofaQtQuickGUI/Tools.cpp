#include "Tools.h"
#include "SofaQtQuickGUI.h"

#include <qqml.h>
#include <QDebug>
#include <QQmlContext>
#include <QQmlEngine>
#include <QFileInfo>
#include <QOpenGLContext>
#include <QOpenGLDebugLogger>

namespace sofa
{

namespace qtquick
{

Tools::Tools(QObject* parent) : QObject(parent)
{

}

Tools::~Tools()
{

}

void Tools::setOverrideCursorShape(int newCursorShape)
{
    if(newCursorShape == overrideCursorShape())
        return;

    QApplication::restoreOverrideCursor();
    if(Qt::ArrowCursor != newCursorShape)
        QApplication::setOverrideCursor(QCursor((Qt::CursorShape) newCursorShape));

    overrideCursorShapeChanged();
}

QQuickWindow* Tools::window(QQuickItem* item) const
{
    if(!item)
        return 0;

    return item->window();
}

void Tools::trimCache(QObject* object)
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

void Tools::clearSettingGroup(const QString& group)
{
	QSettings settings;
	settings.beginGroup(group);
	settings.remove("");
	settings.endGroup();
}

void Tools::setOpenGLDebugContext()
{
    QSurfaceFormat format;
    format.setMajorVersion(4);
    format.setMinorVersion(2);
    format.setProfile(QSurfaceFormat::CompatibilityProfile);
    format.setOption(QSurfaceFormat::DebugContext);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(4);
    QSurfaceFormat::setDefaultFormat(format);
}

void Tools::useOpenGLDebugLogger()
{
    QOpenGLContext *ctx = QOpenGLContext::currentContext();
    if(0 == ctx) {
        qWarning() << "Tools::initializeDebugLogger has been called without a valid opengl context made current";
        return;
    }

    qDebug() << "OpenGL Context" << (QString::number(ctx->format().majorVersion()) + "." + QString::number(ctx->format().minorVersion())).toLatin1().constData();
    qDebug() << "Graphics Card Vendor:" << (char*) glGetString(GL_VENDOR);
    qDebug() << "Graphics Card Model:" << (char*) glGetString(GL_RENDERER);
    qDebug() << "Graphics Card Drivers:" << (char*) glGetString(GL_VERSION);

    if(ctx->hasExtension(QByteArrayLiteral("GL_KHR_debug")))
    {
        QOpenGLDebugLogger* openglDebugLogger = new QOpenGLDebugLogger();
        if(!openglDebugLogger->initialize())
            qDebug() << "OpenGL debug logging disabled: error - the logger could not be initialized";
        else
            qDebug() << "OpenGL debug logging enabled";

        connect(openglDebugLogger, &QOpenGLDebugLogger::messageLogged, [](const QOpenGLDebugMessage &debugMessage) {qDebug() << "OpenGL" << debugMessage.type() << "-" << "Severity;" << debugMessage.severity() << "- Source:" << debugMessage.source() <<  "- Message:" << debugMessage.message();});
        openglDebugLogger->startLogging(QOpenGLDebugLogger::SynchronousLogging);

        connect(ctx, &QOpenGLContext::aboutToBeDestroyed, [=]() {delete openglDebugLogger;});
    }
    else
    {
        qDebug() << "OpenGL debug logging disabled: your graphics card does not support this functionality";
    }
}

void Tools::useDefaultSettingsAtFirstLaunch(const QString& defaultSettingsPath)
{
    QSettings settings;
    bool notFirstLaunch = settings.value("notFirstLaunch", false).toBool();
    if(notFirstLaunch)
        return;

    // copy default.ini into the current directory to be able to open it with QSettings
    QString defaultConfigFilePath = "default.ini";

    QFileInfo fileInfo(defaultConfigFilePath);
    if(!fileInfo.isFile())
    {
        QString finalDefaultSettingsPath = defaultSettingsPath;
        if(finalDefaultSettingsPath.isEmpty())
            finalDefaultSettingsPath = ":/config/default.ini";

        QFile file(finalDefaultSettingsPath);
        if(!file.open(QFile::OpenModeFlag::ReadOnly))
        {
            qDebug() << "ERROR: the default config file has not been found!";
        }
        else
        {
            if(!file.copy(defaultConfigFilePath))
                qDebug() << "ERROR: the config file could not be created!";
            else
                if(!QFile::setPermissions(defaultConfigFilePath, QFile::ReadOwner | QFile::ReadUser | QFile::ReadGroup))
                    qDebug() << "ERROR: cannot set permission on the config file!";
        }
    }

    // copy properties of default.ini into the current settings
    QSettings defaultSettings(defaultConfigFilePath, QSettings::IniFormat);
    copySettings(defaultSettings, settings);

    settings.setValue("notFirstLaunch", true);
}

static void SettingsCopyValuesHelper(const QSettings& src, QSettings& dst)
{
    QStringList keys = src.childKeys();
    foreach(QString key, keys)
        dst.setValue(key, src.value(key));
}

static void SettingsCopyGroupsHelper(const QSettings& src, QSettings& dst)
{
    QStringList groups = src.childGroups();
    foreach(QString group, groups)
    {
        const_cast<QSettings&>(src).beginGroup(group);

        dst.beginGroup(group);
        SettingsCopyGroupsHelper(src, dst);
        dst.endGroup();

        const_cast<QSettings&>(src).endGroup();
    }

    SettingsCopyValuesHelper(src, dst);
}

void Tools::copySettings(const QSettings& src, QSettings& dst)
{
    SettingsCopyGroupsHelper(src, dst);
}

}

}
