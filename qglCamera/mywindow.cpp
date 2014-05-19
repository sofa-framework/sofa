#include "myWindow.h"

myWindow::myWindow(QWidget *parent)
    : myGLWidget(60, parent, "OpenGL et Qt")
{
}

void myWindow::initializeGL()
{
    i=0.0;
    num=0.0;
    x=1.0f;
    y=1.0f;
    z=0.0f;
    glShadeModel(GL_SMOOTH);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);
    glEnable(GL_NORMALIZE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
}

void myWindow::resizeGL(int width, int height)
{
    if(height == 0)
        height = 1;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f, (GLfloat)width/(GLfloat)height, 0.1f, 100.0f);
    glMatrixMode(GL_MODELVIEW);
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
//    glRotatef(num, 1.0, 0.1, 0.1);
//    drawCube();
    glTranslatef(1.5f, 2.0f, -1.0f);
//    drawSphere();

//    light_rotate();
}

using namespace std;
//using namespace qglviewer;

void myWindow::draw()
{
        // Place light at camera position
//        const Vec cameraPos = camera()->position();
//        const GLfloat pos[4] = {cameraPos[0], cameraPos[1], cameraPos[2], 1.0};
//        glLightfv(GL_LIGHT1, GL_POSITION, pos);

        // Orientate light along view direction
//        glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, camera()->viewDirection());

        drawSpiral();
}

// Draws a spiral
void myWindow::drawSpiral()
{
        const float nbSteps = 1000.0;

        glBegin(GL_QUAD_STRIP);
        for (int i=0; i<nbSteps; ++i)
        {
                const float ratio = i/nbSteps;
                const float angle = 21.0*ratio;
                const float c = cos(angle);
                const float s = sin(angle);
                const float r1 = 1.0 - 0.8f*ratio;
                const float r2 = 0.8f - 0.8f*ratio;
                const float alt = ratio - 0.5f;
                const float nor = 0.5f;
                const float up = sqrt(1.0-nor*nor);
                glColor3f(1.0-ratio, 0.2f , ratio);
                glNormal3f(nor*c, up, nor*s);
                glVertex3f(r1*c, alt, r1*s);
                glVertex3f(r2*c, alt+0.05f, r2*s);
        }
        glEnd();
}

void myWindow::init()
{
        // Light setup
        glDisable(GL_LIGHT0);
        glEnable(GL_LIGHT1);

        // Light default parameters
        const GLfloat light_ambient[4]  = {1.0, 1.0, 1.0, 1.0};
        const GLfloat light_specular[4] = {1.0, 1.0, 1.0, 1.0};
        const GLfloat light_diffuse[4]  = {1.0, 1.0, 1.0, 1.0};

        glLightf( GL_LIGHT1, GL_SPOT_EXPONENT, 3.0);
        glLightf( GL_LIGHT1, GL_SPOT_CUTOFF,   10.0);
        glLightf( GL_LIGHT1, GL_CONSTANT_ATTENUATION,  0.1f);
        glLightf( GL_LIGHT1, GL_LINEAR_ATTENUATION,    0.3f);
        glLightf( GL_LIGHT1, GL_QUADRATIC_ATTENUATION, 0.3f);
        glLightfv(GL_LIGHT1, GL_AMBIENT,  light_ambient);
        glLightfv(GL_LIGHT1, GL_SPECULAR, light_specular);
        glLightfv(GL_LIGHT1, GL_DIFFUSE,  light_diffuse);

}

//QString myWindow::helpString() const
//{
//        QString text("<h2>C a m e r a L i g h t</h2>");
//        text += "See the <b>Mouse</b> tab and the documentation web pages for details.<br><br>";
//        text += "Press <b>Escape</b> to exit the viewer.";
//        return text;
//}
