#include <GL/glew.h>
#include "Window.h"

#include <qqml.h>
#include <QDebug>
#include <QQmlContext>
#include <QQmlEngine>
#include <QSettings>

static int registerType = qmlRegisterType<Window>("Window", 1, 0, "Window");

Window::Window(QWindow* parent) : QQuickWindow(parent)
{
    // since we draw our scene in a fbo it is not useful anymore, let qt clear the render buffer for us
    setClearBeforeRendering(false);

	connect(this, &Window::sceneGraphInitialized, &glewInit);
}

Window::~Window()
{
	clearSettingGroup("dummy");
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

void Window::trimCache(QObject* object)
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

void Window::clearSettingGroup(const QString& group)
{
	QSettings settings;
	settings.beginGroup(group);
	settings.remove("");
	settings.endGroup();
}