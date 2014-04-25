#include "QtViewer.h"
#include <GL/glut.h>

//GLfloat light_position[] = { 0.0, 0.0, 25.0, 0.0 };
//GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
//GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
//GLfloat light_specular[] = { 1.0, 1.0, 1.0, 0.0 };

GLfloat camera_position[] = { 0.0, 0.0, 15.0, 0.0 };
GLfloat znear = camera_position[2]-10;
GLfloat zfar = camera_position[2]+10;


QtViewer::QtViewer(QGLWidget *parent) :
    QGLWidget(parent)
{
}

void QtViewer::initializeGL()
{
    glClearColor (0.0, 0.0, 0.0, 0.0);

//    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
//    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
//    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
//    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

//    glDisable(GL_LIGHTING);
//    glEnable(GL_LIGHT0);

    glEnable(GL_DEPTH_TEST);

}

void QtViewer::paintGL()
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity ();


    gluLookAt ( camera_position[0],camera_position[1],camera_position[2], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

//    sofaScene.glDraw();

    // display a box, for debug
    glColor3f (1.0, 0.0, 0.0);
    glutWireCube (1.0);

}

void QtViewer::resizeGL(int w, int h)
{
    glViewport (0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    gluPerspective (55.0, (GLfloat) w/(GLfloat) h, znear, zfar );
    glMatrixMode (GL_MODELVIEW);
}


