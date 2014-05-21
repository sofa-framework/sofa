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

    Q_INVOKABLE void saveScreenshot(const QString& imagePath = QString());

public slots:
    void paint();
    void grab();
    void cleanup();

protected:
    void createFrameBuffer();

private slots:


private:
    QOpenGLFramebufferObject*   myFramebuffer;
    QOpenGLShaderProgram*       myCompositionShaderProgram;

};

#endif // WINDOW_H
