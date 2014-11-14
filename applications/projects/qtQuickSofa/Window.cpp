#include "Window.h"
#include <QTime>
#include <QOpenGLContext>
#include <QtGui/QImage>

Window::Window(QWindow* parent) : QQuickWindow(parent)
{
    // since we draw our scene in a fbo it is not useful anymore, let qt clear the render buffer for us
    setClearBeforeRendering(false);
}

Window::~Window()
{
	
}
