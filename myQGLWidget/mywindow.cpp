#include "myWindow.h"

myWindow::myWindow(QWidget *parent)
    : myGLWidget(60, parent, "Premier Polygone avec OpenGL et Qt")
{
}

void myWindow::initializeGL()
{
    i=0.0;
    num=0.0;
    x=1.0f;
    y=1.0f;
    z=0.0f;
    glShadeModel(GL_SMOOTH);//GL_FLAT or GL_SMOOTH shading
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);// specify the clear value for the depth buffer
//    glEnable(GL_LIGHTING);
//    glEnable(GL_LIGHT0);
//    glEnable(GL_LIGHT1);
//    glEnable(GL_LIGHT2);
    glEnable(GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);//Passes if the incoming depth value is less than or equal to the stored depth value
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);//GL_NICEST:The most correct, or highest quality, option should be chosen
//GL_PERSPECTIVE_CORRECTION_HINT: Indicates the quality of color, texture coordinate, and fog coordinate interpolation
}

void myWindow::resizeGL(int width, int height)
{
    if(height == 0)
        height = 1;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);//Applies subsequent matrix operations to the projection matrix stack
    glLoadIdentity();
    gluPerspective(45.0f, (GLfloat)width/(GLfloat)height, 0.1f, 100.0f);//set up a perspective projection matrix
    glMatrixMode(GL_MODELVIEW);//Applies subsequent matrix operations to the modelview matrix stack
    glLoadIdentity();
    glPushMatrix();
}

void myWindow::paintGL()
{
    i=1.0f;
    num+=1.9;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glTranslatef(-1.5f, 0.0f, -10.0f);
    drawSphere();
    glTranslatef(1.5f, 2.0f, -1.0f);
    glRotatef(-num, 1.0, 0.1, -0.1);
    drawCube();

//    glRotatef(-num, 1.0, -0.1, -0.1);


//    light_rotate();
//    light_stop_rotate();
//    drawSquare();

//    glTranslatef(3.0f, 0.0f, -6.0f);
//    drawTriangle();
//    glTranslatef(-1.5f, 0.0f, -10.0f);
//drawPoints();
//drawDroite();
//glutMainLoop ();
//    drawGrid();
}
void myWindow::drawPoints()
{
    glBegin(GL_POINTS);
    for (GLint j = -10; j<10; ++j)
    {
        glVertex3f(j,0,2);
    }
    glEnd();
}
void myWindow::drawDroite()
{
    glBegin(GL_LINES);
    for (GLint toto =-10;toto<20;++toto)
    {
        glColor3f(.6,.3,.6);
        glVertex3i(toto,0,1);
        glVertex3i(toto+1,0,1);
    }
    glEnd();
}
void myWindow::drawSphere()
{
    float a = 9;
    float b = 10;
    float da = ( M_PI / a );
    float db = ( 2.0f * M_PI / b );
    glBegin(GL_QUADS);
    glColor3f(1.0,0.0,0.0);
    glNormal3f(0,0,1);
    for( int i = 0; i < a + 1 ; i++ )
    {
     float r0 = sin ( i * da );
     float y0 = cos ( i * da );
     float r1 = sin ( (i+1) * da );
     float y1 = cos ( (i+1) * da );
     for( int j = 0; j < b + 1 ; j++ ) {
      float x0 = r0 * sin( j * db );
      float z0 = r0 * cos( j * db );
      float x1 = r0 * sin( (j+1) * db );
      float z1 = r0 * cos( (j+1) * db );

      float x2 = r1 * sin( j * db );
      float z2 = r1 * cos( j * db );
      float x3 = r1 * sin( (j+1) * db );
      float z3 = r1 * cos( (j+1) * db );

//    glColor3f((x0+x1+x2+x3)/4.0,(y0+y1)/2.0,(z0+z1+z2+z3)/4.0);
//      glEnable(GL_NORMALIZE);
      glVertex3f(x0,y0,z0);
      glNormal3f((x0-x1)*(x3-x1),(y0-y0)*(y1-y0),(z0-z1)*(z3-z1));
      glVertex3f(x1,y0,z1);
//      glNormal3f((x0-x3)*(x2-x3),(y0-y1)*(y1-y1),(z0-z3)*(z2-z3));
      glVertex3f(x3,y1,z3);
      glVertex3f(x2,y1,z2);
//      glNormal3f((x0-x2)*(x3-x2),(y0-y1)*(y1-y1),(z0-z2)*(z3-z2));
  }

    }
glEnd();
}
void myWindow::drawSquare()
{
//    glBegin(GL_QUADS);
//    glColor3f(1.0f, 1.0f, 0.0f);
//    glVertex3f(-1.0f, 1.0f, 0.0f);
//    glVertex3f(-1.0f, -1.0f, 0.0f);
//    glVertex3f(1.0f, -1.0f, 0.0f);
//    glVertex3d(1.0f, 1.0f, 0.0f);
//    glEnd();
    glBegin(GL_QUADS);
    glColor3f(1.0f, 1.0f, 0.0f);
    glVertex3f(-x, y, z);
    glVertex3f(-x, -y, z);
    glVertex3f(x, -y, z);
    glVertex3d(x, y, z);
    glEnd();
}
void myWindow::drawCube()
{
    glBegin(GL_QUADS);
       // Face Avant
       glNormal3f(0.0f,0.0f,1.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
       // Face Arrière
       glNormal3f(0.0f,0.0f,-1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
       // Face Haut
       glNormal3f(0.0f,1.0f,0.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
       // Face Bas
       glNormal3f(0.0f,-1.0f,0.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
       // Face Droite
       glNormal3f(1.0f,0.0f,0.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
       // Face Gauche
       glNormal3f(-1.0f,0.0f,0.0f);
       glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);
       glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
       glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);
       glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);
   glEnd();
}
void myWindow::drawTriangle()
{
    glBegin(GL_TRIANGLES);
    glColor3f(1.0f, .0f, 1.0f);
    glVertex3f(0.0f, 1.0f, 0.0f);
    glVertex3f(-1.0f, -1.0f, 0.0f);
    glVertex3f(1.0f, -1.0f, 0.0f);
    glEnd();
}
void myWindow::drawGrid()
{
    glBegin(GL_QUADS);
    glVertex3f( 0,-0.001, 0);
    glVertex3f( 0,-0.001,10);
    glVertex3f(10,-0.001,10);
    glVertex3f(10,-0.001, 0);
    glEnd();
    glBegin(GL_LINES);
    for(int i=0;i<=10;i++) {
        if (i==0) { glColor3f(.6,.3,.3); } else { glColor3f(.25,.25,.25); };
            glVertex3f(i,0,0);
            glVertex3f(i,0,10);
        if (i==0) { glColor3f(.3,.3,.6); } else { glColor3f(.25,.25,.25); };
        glVertex3f(0,0,i);
        glVertex3f(10,0,i);
    };
    glEnd();
}
void myWindow::light_rotate()
{
//    glTranslatef(-1.5f, 0.0f, -10.0f);
    glRotatef(num, 0.1, 0.1, 0.1);
    drawSquare();
}
void myWindow::light_stop_rotate()
{
glRotatef(num, 0.0, 0.0, 0.0);
}
void myWindow::light0_activate()
{
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    GLfloat lightColor0[] = {1.5f, 0.5f, 0.5f, 0.1f}; //Color (0.5, 0.5, 0.5)
    GLfloat lightPos0[] = {4.0f, 0.0f, 8.0f, 1.0f}; //Positioned at (4, 0, 8)
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
    glLightfv(GL_LIGHT0, GL_AMBIENT, lightColor0);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, lightColor0);
    glLightfv(GL_LIGHT0, GL_SPECULAR, lightColor0);
}
void myWindow::light0_desactivate()
{
    glDisable(GL_LIGHT0);
}
void myWindow::light1_activate()
{
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT1);
    GLfloat lightColor1[] = {1.1f, 1.0f, 0.1f, 0.1f};
    GLfloat lightPos1[] = {1.0f, 0.0f, 1.0f, 1.0f};
    glLightfv(GL_LIGHT1, GL_AMBIENT, lightColor1);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, lightColor1);
    glLightfv(GL_LIGHT1, GL_SPECULAR, lightColor1);
    glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);glEnable(GL_LIGHT1);
}
void myWindow::light1_desactivate()
{
    glDisable(GL_LIGHT1);
}
void myWindow::light0_moving()
{
    GLfloat lightPos0[] = {i/10.0f+4.0f, i/10.0f+0.0f, i/10.0f+8.0f, 1.0f}; //Positioned at (4, 0, 8)
    glLightfv(GL_LIGHT0, GL_POSITION, lightPos0);
//    glLightf(GL_LIGHT1, GL_POSITION, (i,i,i,i));
}
void myWindow::light1_moving()
{
    GLfloat lightPos1[] = {i/10.0f+1.0f, 0.0f, 1.0f, 1.0f};
    glLightfv(GL_LIGHT1, GL_POSITION, lightPos1);
}
void myWindow::light2_activate()
{
    glEnable(GL_LIGHTING);
    GLfloat lightColor2[] = {0.5f, 1.0f, 0.1f, 0.1f};
    GLfloat lightPos2[] = {2.0f, 2.0f, 2.0f, 2.0f};
    glLightfv(GL_LIGHT2, GL_AMBIENT, lightColor2);
    glLightfv(GL_LIGHT2, GL_DIFFUSE, lightColor2);
    glLightfv(GL_LIGHT2, GL_SPECULAR, lightColor2);
    glLightfv(GL_LIGHT2, GL_POSITION, lightPos2);
    glEnable(GL_LIGHT2);
}
void myWindow::light2_desactivate()
{
    glDisable(GL_LIGHT2);
}
void myWindow::light2_moving()
{
    GLfloat lightPos2[] = {i+2.0f, 2.0f, i+2.0f, 2.0f};
    glLightfv(GL_LIGHT2, GL_POSITION, lightPos2);
}
void myWindow::keyPressEvent(QKeyEvent *keyEvent)
{
    switch(keyEvent->key())
    {
        case Qt::Key_Escape:
            close();
            break;
        case Qt::Key_W:
            light0_activate();
            break;
        case Qt::Key_X:
            light0_desactivate();
            break;
        case Qt::Key_C:
            light1_activate();
            break;
        case Qt::Key_V:
            light1_desactivate();
            break;
        case Qt::Key_Left:
            i+=0.1;
            light0_moving();
            break;
        case Qt::Key_Right:
            i+=-0.1;
            light0_moving();
            break;
        case Qt::Key_Up:
            i+=0.1;
            light1_moving();
            break;
        case Qt::Key_Down:
            i+=-0.1;
            light1_moving();
            break;
        case Qt::Key_1:
            i+=0.1;
            light2_moving();
            break;
        case Qt::Key_2:
            i-=0.1;
            light2_moving();
            break;
        case Qt::Key_S:
            light2_activate();
            break;
        case Qt::Key_D:
            light2_desactivate();
            break;

    }
}
//http://rvirtual.free.fr/programmation/OpenGl/Eclairage.htm
//http://www.glprogramming.com/red/chapter05.html
//http://fr.wikipedia.org/wiki/Ombrage_Phong
//http://www.videotutorialsrock.com/opengl_tutorial/lighting/text.php
//http://www.qtfr.org/viewtopic.php?id=14637
//http://cpp.developpez.com/redaction/data/pages/users/gbdivers/qtopengl/
