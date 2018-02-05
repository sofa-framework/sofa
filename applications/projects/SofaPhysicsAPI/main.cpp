/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, v17.12                  *
*                (c) 2006-2017 INRIA, USTL, UJF, CNRS, MGH                    *
*                                                                             *
* This program is free software; you can redistribute it and/or modify it     *
* under the terms of the GNU General Public License as published by the Free  *
* Software Foundation; either version 2 of the License, or (at your option)   *
* any later version.                                                          *
*                                                                             *
* This program is distributed in the hope that it will be useful, but WITHOUT *
* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or       *
* FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for    *
* more details.                                                               *
*                                                                             *
* You should have received a copy of the GNU General Public License along     *
* with this program. If not, see <http://www.gnu.org/licenses/>.              *
*******************************************************************************
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include "SofaPhysicsAPI.h"
#include <sofa/helper/system/glut.h>

#include <stdio.h>
#include <stdlib.h>

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
    const char* defaultScene="xml/newEye.scn";
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
    glutInitWindowSize(720,720);
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
static bool rendering_multiviews = false;

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
        if (rendering_multiviews)
            glViewport(0, glut_height - glut_height/2, glut_width/2, glut_height/2);
        mainSimulation->drawGL();

        unsigned int nbDataControllers = mainSimulation->getNbDataControllers();
        SofaPhysicsDataController** dataControllers = mainSimulation->getDataControllers();
        for (unsigned int i=0; i<nbDataControllers; ++i)
        {
            SofaPhysicsDataController* m = dataControllers[i];
            m->setValue("30");
            printf("DataController[%i]\n applied", i);
        }

        unsigned int nbDataMonitors = mainSimulation->getNbDataMonitors();
        SofaPhysicsDataMonitor** dataMonitors = mainSimulation->getDataMonitors();
        for (unsigned int i=0; i<nbDataMonitors; ++i)
        {
            SofaPhysicsDataMonitor* m = dataMonitors[i];
            printf("DataMonitor[%i] = %s\n", i, m->getValue());
        }

        unsigned int nbMeshes = mainSimulation->getNbOutputMeshes();
        SofaPhysicsOutputMesh** meshes = mainSimulation->getOutputMeshes();

        // first compute bbox
        float bbmin[3] = { 0.0f, 0.0f, 0.0f };
        float bbmax[3] = { 0.0f, 0.0f, 0.0f };
        float bbcenter[3] = { 0.0f, 0.0f, 0.0f };
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
        for (unsigned int c=0; c<3; ++c)
            bbcenter[c] = (bbmin[c]+bbmax[c])*0.5f;
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
                printf("  mesh %3d: %6d points%s%s, name = \"%s\"\n", i, m->getNbVertices(), (m->getVTexCoords() ? " with UVs" : ""), (m->getVNormals() ? " with normals" : ""), m->getName());
            }
        }
        ++counter;

        if (rendering_multiviews)
        {
            glViewport(0, 0, glut_width, glut_height);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_LIGHTING);
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            // compute the scales applied on each axis, by keeping the smallest per-pixel one
            float xwpixelscale = (glut_width*0.5f)/(bbmax[0]-bbmin[0]);
            float ywpixelscale = (glut_width*0.5f)/(bbmax[1]-bbmin[1]);
            float yhpixelscale = (glut_height*0.5f)/(bbmax[1]-bbmin[1]);
            float zhpixelscale = (glut_height*0.5f)/(bbmax[2]-bbmin[2]);

            float pixelscale = xwpixelscale;
            if (ywpixelscale<pixelscale) pixelscale = ywpixelscale;
            if (yhpixelscale<pixelscale) pixelscale = yhpixelscale;
            if (zhpixelscale<pixelscale) pixelscale = zhpixelscale;

            float wscale = pixelscale/(glut_width*0.5f);
            float hscale = pixelscale/(glut_height*0.5f);

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
                    Real vx = (vpos[v*3+0] - bbcenter[0]);
                    Real vy = (vpos[v*3+1] - bbcenter[1]);
                    Real vz = (vpos[v*3+2] - bbcenter[2]);
                    glVertex2f( 0.5f+vx*wscale, 0.5f+vy*hscale);
                    glVertex2f( 0.5f+vx*wscale,-0.5f+vz*hscale);
                    glVertex2f(-0.5f+vy*wscale,-0.5f+vz*hscale);
                }
            }
            glEnd();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
            glPopMatrix();
            glEnable(GL_DEPTH_TEST);
        }
    }
    glutSwapBuffers();
}

static void glut_reshape(int w, int h)
{
    glut_width = w;
    glut_height = h;
    glutPostRedisplay();
}

static void glut_keyboard(unsigned char k, int /*x*/, int /*y*/)
{
    printf("keyboard %d -> %c\n",(int)k,(char)k);
    switch (k)
    {
    case ' ':
        mainSimulation->setAnimated(!mainSimulation->isAnimated());
        break;
    case '0':
        mainSimulation->reset();
        break;
    case '1':
        mainSimulation->step();
        printf("simulation time = %6f\n", mainSimulation->getTime());
        break;
    case 13:
        rendering_multiviews = !rendering_multiviews;
        printf("rendering mode: %s\n", (rendering_multiviews ? "multiviews" : "singleview"));
        break;
    case 27:
        printf("exit\n");
        exit(0);
        break;
    }
    glutPostRedisplay();
}

static void glut_mouse(int /*button*/, int /*state*/, int /*x*/, int /*y*/)
{
}

static void glut_motion(int /*x*/, int /*y*/)
{
}

static void glut_special(int k, int /*x*/, int /*y*/)
{
    printf("special %d\n",(int)k);
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
