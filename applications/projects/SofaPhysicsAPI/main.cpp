#include "SofaPhysicsAPI.h"
#include <sofa/helper/system/glut.h>

#include <stdio.h>

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
    const char* defaultScene="Demos/caduceus.scn";
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
    glViewport(0, 0, glut_width, glut_height);
    static float color = 0.0f;
    //color += 0.01; while(color > 1) color -= 2;
    float absc = (color < 0.0f) ? -color : color;
    glClearColor(0.0f,0.0f,absc,1.0f);
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    if (mainSimulation)
    {
        glViewport(0, glut_height - glut_height/2, glut_width/2, glut_height/2);
        mainSimulation->drawGL();

        unsigned int nbMeshes = mainSimulation->getNbOutputMeshes();
        SofaPhysicsOutputMesh** meshes = mainSimulation->getOutputMeshes();

        // first compute bbox
        float bbmin[3] = { 0.0f, 0.0f, 0.0f };
        float bbmax[3] = { 0.0f, 0.0f, 0.0f };
        unsigned int totalpoints = 0;
        for (unsigned int i=0; i<nbMeshes; ++i)
        {
            SofaPhysicsOutputMesh* m = meshes[i];
            unsigned int nbv = m->getNbVertices();
            if (!nbv) continue;
            const Real* vpos = m->getVPositions();
            if (!vpos) continue;
            if (!totalpoints)
            {
                bbmin[0] = bbmax[0] = vpos[0];
                bbmin[1] = bbmax[1] = vpos[1];
                bbmin[2] = bbmax[2] = vpos[2];
            }
            for (unsigned int v=0; v<nbv; ++v)
                for (unsigned int c=0; c<3; ++c)
                {
                    Real r = vpos[v*3+c];
                    if (r < bbmin[c]) bbmin[c] = r;
                    else if (r > bbmax[c]) bbmax[c] = r;
                }
            totalpoints += nbv;
        }
        static int counter = 0;
        if (!(counter%1000))
        {
            printf("%3d meshes, %6d total points, bbox = < %6f, %6f, %6f > - < %6f, %6f, %6f >\n",nbMeshes,totalpoints,bbmin[0],bbmin[1],bbmin[2],bbmax[0],bbmax[1],bbmax[2]);
        }
        if (!counter)
        {
            for (unsigned int i=0; i<nbMeshes; ++i)
            {
                SofaPhysicsOutputMesh* m = meshes[i];
                printf("  mesh %3d: %6d points, name = \"%s\"\n",i,m->getNbVertices(),m->getName());
            }
        }
        ++counter;

        glViewport(0, 0, glut_width, glut_height);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();

        // Then display the points
        glBegin(GL_POINTS);
        for (unsigned int i=0; i<nbMeshes; ++i)
        {
            SofaPhysicsOutputMesh* m = meshes[i];
            unsigned int nbv = m->getNbVertices();
            if (!nbv) continue;
            const Real* vpos = m->getVPositions();
            if (!vpos) continue;

            //int colorindex = 1+((int)(m->getName()[0])%7);
            int colorindex = 1+((int)(i)%7);
            glColor3f((float)(colorindex&1), (float)((colorindex>>1)&1), (float)((colorindex>>2)&1));

            for (unsigned int v=0; v<nbv; ++v)
            {
                Real vx = (vpos[v*3+0] - bbmin[0]) / (bbmax[0] - bbmin[0]);
                Real vy = (vpos[v*3+1] - bbmin[1]) / (bbmax[1] - bbmin[1]);
                Real vz = (vpos[v*3+2] - bbmin[2]) / (bbmax[2] - bbmin[2]);
                glVertex2f( vx, vy);
                glVertex2f( vx,-vz);
                glVertex2f(-vy,-vz);
            }
        }
        glEnd();
        glMatrixMode(GL_PROJECTION);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        glEnable(GL_DEPTH_TEST);

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
    switch (k)
    {
    case ' ':
        mainSimulation->setAnimated(!mainSimulation->isAnimated());
        break;
    case '0':
        mainSimulation->reset();
        break;
    }
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
    if (mainSimulation && mainSimulation->isAnimated())
    {
        mainSimulation->step();

        // update FPS on window title
        static double lastFPS = 0.0;
        double currentFPS = mainSimulation->getCurrentFPS();
        if (currentFPS != lastFPS)
        {
            lastFPS = currentFPS;
            char buf[1000];
            const char* sceneFileName = mainSimulation->getSceneFileName();
            sprintf(buf, "SOFA :: %s :: %.1f FPS", sceneFileName, currentFPS);
            glutSetWindowTitle(buf);
        }

        // update rendered view
        glutPostRedisplay();
    }
}
