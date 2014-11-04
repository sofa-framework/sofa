#ifndef WINDOW_H
#define WINDOW_H

#include <QObject>
#include <QQuickWindow>
#include <QtGui/QOpenGLFramebufferObject>
#include <QtGui/QOpenGLShaderProgram>

class Window : public QQuickWindow
{
    Q_OBJECT

public:
    explicit Window(QWindow* parent = 0);
    ~Window();

};

#endif // WINDOW_H
