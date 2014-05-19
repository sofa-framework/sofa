#ifndef MYWINDOW_H
#define MYWINDOW_H

#include <QtGui/QMainWindow>
#include "myGLWidget.h"
#include <QtOpenGL>
#include <QGLWidget>
#include <qgl.h>
//#include <GL/glut.h>

class myWindow : public myGLWidget
{
    Q_OBJECT
public:
    explicit myWindow(QWidget *parent = 0);
    void initializeGL();
    void resizeGL(int width, int height);
    void paintGL();
    void light0_activate();
    void light0_desactivate();
    virtual void keyPressEvent( QKeyEvent *keyEvent );
    float num;
    void light1_activate();
    void light1_desactivate();
    GLfloat i;
    GLfloat x, y, z;
    void light0_moving();
    void drawSphere();
    void drawSquare();
    void drawTriangle();
    void drawGrid();
    void light_rotate();
    void myWindow::drawCube();
    void light1_moving();
    void light_stop_rotate();
    void light2_activate();
    void light2_desactivate();
    void drawPoints();
    void drawDroite();
    void light2_moving();
};

#endif // MYWINDOW_H
