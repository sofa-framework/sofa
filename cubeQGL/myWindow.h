#ifndef MYWINDOW_H
#define MYWINDOW_H

#include <QImage>
#include "myGLWidget.h"


class myWindow : public myGLWidget
{
    Q_OBJECT
public:
    explicit myWindow(QWidget *parent = 0);
    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void loadTexture(QString textureName);
private :
    GLuint texture[1];
    float f_x;
};

#endif // MYWINDOW_H
