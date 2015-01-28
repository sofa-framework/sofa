#include <GL/glew.h>
#include "Window.h"

#include <qqml.h>
#include <QDebug>
#include <QQmlContext>
#include <QQmlEngine>
#include <QSettings>
//#include <QOpenGLContext>

Window::Window(QWindow* parent) : QQuickWindow(parent)
{
    // since we draw our scene in a fbo it is not useful anymore, let qt clear the render buffer for us
    setClearBeforeRendering(false);

// 	QSurfaceFormat format;
// 	format.setMajorVersion(3);
// 	format.setMajorVersion(2);
// 	format.setProfile(QSurfaceFormat::OpenGLContextProfile::CompatibilityProfile);
// 	setFormat(format);

	connect(this, &Window::sceneGraphInitialized, &glewInit);
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