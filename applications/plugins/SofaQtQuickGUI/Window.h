#ifndef WINDOW_H
#define WINDOW_H

#include <QQuickWindow>

class QOpenGLDebugLogger;

class Window : public QQuickWindow
{
    Q_OBJECT

public:
    explicit Window(QWindow* parent = 0);
    ~Window();

private:
    void initialize();
    void invalidate();

private:
    QOpenGLDebugLogger* myOpenglDebugLogger;

};

#endif // WINDOW_H
