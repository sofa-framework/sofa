#ifndef WINDOW_H
#define WINDOW_H

#include <QGLWidget>

class Window : public QGLWidget
{
public:
    Window(QGLWidget *parent = 0);
    ~Window();
    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void loadTexture(QString textureName);
    void InitGL();
    void affichage();
private :
    GLuint texture[1];
    float f_x;
};

#endif // WINDOW_H
