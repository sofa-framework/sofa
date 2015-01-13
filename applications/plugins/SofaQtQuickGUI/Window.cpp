#include "Window.h"

#include <qqml.h>
#include <QDebug>
#include <QQmlContext>
#include <QQmlEngine>
#include <QSettings>

#include <QOpenGLContext>
#include <QOpenGLDebugLogger>

Window::Window(QWindow* parent) : QQuickWindow(parent),
    myOpenglDebugLogger(0)
{
    QSurfaceFormat format;
    format.setMajorVersion(4);
    format.setMinorVersion(2);
    format.setProfile(QSurfaceFormat::CompatibilityProfile);
    format.setOption(QSurfaceFormat::DebugContext);
    format.setDepthBufferSize(24);
    format.setStencilBufferSize(8);
    format.setSamples(4);
    setFormat(format);

    connect(this, &Window::sceneGraphInitialized, this, &Window::initialize);
    connect(this, &Window::sceneGraphInvalidated, this, &Window::invalidate);
}

Window::~Window()
{
	
}

void Window::setOverrideCursorShape(int newCursorShape)
{
	if(newCursorShape == overrideCursorShape())
		return;

	QApplication::restoreOverrideCursor();
	if(Qt::ArrowCursor != newCursorShape)
		QApplication::setOverrideCursor(QCursor((Qt::CursorShape) newCursorShape));

	overrideCursorShapeChanged();
}

void Window::initialize()
{
    QOpenGLContext *ctx = QOpenGLContext::currentContext();
    qDebug() << "OpenGL Context" << (QString::number(ctx->format().majorVersion()) + "." + QString::number(ctx->format().minorVersion())).toLatin1().constData();
    qDebug() << "Graphics Card Vendor:" << (char*) glGetString(GL_VENDOR);
    qDebug() << "Graphics Card Model:" << (char*) glGetString(GL_RENDERER);
    qDebug() << "Graphics Card Drivers:" << (char*) glGetString(GL_VERSION);

    if(ctx->hasExtension(QByteArrayLiteral("GL_KHR_debug")))
    {
        myOpenglDebugLogger = new QOpenGLDebugLogger(this);
        if(!myOpenglDebugLogger->initialize())
            qDebug() << "OpenGL debug logging disabled: error - the logger could not be initialized";
        else
            qDebug() << "OpenGL debug logging enabled";

        connect(myOpenglDebugLogger, &QOpenGLDebugLogger::messageLogged, [](const QOpenGLDebugMessage &debugMessage) {qDebug() << "OpenGL" << debugMessage.type() << "-" << "Severity;" << debugMessage.severity() << "- Source:" << debugMessage.source() <<  "- Message:" << debugMessage.message();});
        myOpenglDebugLogger->startLogging(QOpenGLDebugLogger::SynchronousLogging);
    }
    else
    {
        qDebug() << "OpenGL debug logging disabled: your graphics card does not support this functionality";
    }
}

void Window::invalidate()
{
    delete myOpenglDebugLogger;
    myOpenglDebugLogger = 0;
}
