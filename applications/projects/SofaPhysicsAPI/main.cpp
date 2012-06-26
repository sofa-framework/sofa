#include "SofaPhysicsSimulation.h"
#include <sofa/helper/system/glut.h>

#include <iostream>

std::string mainProcessName;

SofaPhysicsSimulation* mainSimulation = NULL;

// glut callbacks

static void glut_create();
static void glut_display();
static void glut_reshape(int w, int h);
static void glut_keyboard(unsigned char k, int x, int y);
static void glut_mouse(int button, int state, int x, int y);
static void glut_motion(int x, int y);
static void glut_special(int k, int x, int y);
static void glut_idle();



int main(int argc, char *argv[])
{
    mainProcessName = argv[0];

    const char* defaultScene="";
    if (argc > 1)
    {
        defaultScene = argv[1];
    }


    glutInit(&argc, argv);
    glut_create();

    mainSimulation = new SofaPhysicsSimulation;
    if (defaultScene && *defaultScene)
    {
        if (!mainSimulation->load(defaultScene))
        {
            delete mainSimulation;
            return 1;
        }
        if (!mainSimulation->isAnimated())
            mainSimulation->start();
    }

    glutMainLoop();
    return 0;
}

static void glut_create()
{
    glutInitDisplayMode ( GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE );
    glutCreateWindow ( ":: SOFA ::" );

    glClearColor ( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glutSwapBuffers ();
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glutSwapBuffers ();

    glutReshapeFunc ( glut_reshape );
    glutIdleFunc ( glut_idle );
    glutDisplayFunc ( glut_display );
    glutKeyboardFunc ( glut_keyboard );
    glutSpecialFunc ( glut_special );
    glutMouseFunc ( glut_mouse );
    glutMotionFunc ( glut_motion );
    glutPassiveMotionFunc ( glut_motion );

}

static int glut_width = 0;
static int glut_height = 0;

static void glut_display()
{
    if (glut_width && glut_height)
        glViewport(0, glut_height - glut_height/2, glut_width/2, glut_height/2);
    static float color = 0.0f;
    color += 0.01;
    while(color > 1) color -= 2;
    glClearColor(0.0f,0.0f,fabs(color),1.0f);
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    if (mainSimulation)
    {
        mainSimulation->drawGL();
    }
    glutSwapBuffers();
}

static void glut_reshape(int w, int h)
{
    glut_width = w;
    glut_height = h;
}

static void glut_keyboard(unsigned char k, int x, int y)
{
}

static void glut_mouse(int button, int state, int x, int y)
{
}

static void glut_motion(int x, int y)
{
}

static void glut_special(int k, int x, int y)
{
}

static void glut_idle()
{
    if (mainSimulation)
    {
        if (mainSimulation->isAnimated())
        {
            mainSimulation->step();

            // update FPS on window title
            static double lastFPS = 0.0;
            double currentFPS = mainSimulation->getCurrentFPS();
            if (currentFPS != lastFPS)
            {
                lastFPS = currentFPS;
                char buf[100];
                sprintf(buf, "%.1f FPS", currentFPS);
                std::string title = "SOFA";
                std::string sceneFileName = mainSimulation->getSceneFileName();
                if (!sceneFileName.empty())
                {
                    title += " :: ";
                    title += sceneFileName;
                }
                title += " :: ";
                title += buf;
                glutSetWindowTitle(title.c_str());
            }

            // update rendered view
            glutPostRedisplay();
        }
    }
}
