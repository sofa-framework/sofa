/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 RC 1        *
*                (c) 2006-2011 INRIA, USTL, UJF, CNRS, MGH                    *
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
* with this program; if not, write to the Free Software Foundation, Inc., 51  *
* Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.                   *
*******************************************************************************
*                            SOFA :: Applications                             *
*                                                                             *
* Authors: The SOFA Team and external contributors (see Authors.txt)          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <iostream>
#include <fstream>
#include <ctime>
#include <GL/glew.h>
#include <GL/glut.h>
using std::cerr;
using std::endl;
using std::cout;

#include <sofa/helper/ArgumentParser.h>

#include <sofa/helper/system/PluginManager.h>
#include <sofa/component/init.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/component/misc/WriteState.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/core/visual/VisualParams.h>

/** Prototype used to completely encapsulate the use of Sofa in an OpenGL application, without any standard Sofa GUI.
 *
 * @author Francois Faure, 2014
 * */
struct SofaScene : public sofa::simulation::graph::DAGSimulation
{
    typedef sofa::simulation::graph::DAGSimulation Parent;

    sofa::simulation::Node::SPtr groot;
    sofa::core::visual::DrawToolGL   drawToolGL;
    sofa::core::visual::VisualParams* vparams;
    bool debug;

    SofaScene(): debug(true) {}

    void init( std::vector<std::string>& plugins, const std::string& fileName )
    {
        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

        sofa::component::init();
        sofa::simulation::xml::initXml();

        vparams = sofa::core::visual::VisualParams::defaultInstance();
        vparams->drawTool() = &drawToolGL;


        // --- plugins ---
        for (unsigned int i=0; i<plugins.size(); i++)
            sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);

        sofa::helper::system::PluginManager::getInstance().init();


        // --- Create simulation graph ---
        groot = sofa::simulation::getSimulation()->load(fileName.c_str());
        if (groot==NULL)
        {
            groot = sofa::simulation::getSimulation()->createNewGraph("");
        }

        Parent::init(groot.get());

        if( debug ){
            cerr<<"scene loaded" << endl;
            sofa::simulation::getSimulation()->print(groot.get());
        }

    }

    void glDraw()
    {
        sofa::simulation::getSimulation()->draw(vparams,groot.get());
    }

    void animate()
    {
        sofa::simulation::getSimulation()->animate(groot.get(),0.04);
    }
};


// ---------------------------------------------------------------------
// --- A basic glut application featuring a Sofa simulation
// Sofa is invoked only through variable sofaScene.
// ---------------------------------------------------------------------

SofaScene sofaScene;

GLfloat light_position[] = { 0.0, 0.0, 15.0, 0.0 };
GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat light_specular[] = { 1.0, 1.0, 1.0, 0.0 };


void init(void)
{
    glClearColor (0.0, 0.0, 0.0, 0.0);

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
    glLightfv(GL_LIGHT0, GL_POSITION, light_position);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);
}


void display(void)
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity ();

    gluLookAt ( light_position[0],light_position[1],light_position[2], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    sofaScene.glDraw();

    // display a box, for debug
    glScalef (1.0, 2.0, 1.0);
    glColor3f (1.0, 0.0, 0.0);
    glutWireCube (1.0);
    glutSwapBuffers();

    // Due to some bug, the first display displays nothing. Hence this poor fix:
    static int first = true;
    if( first ) { glutPostRedisplay(); first = false; }
}

void reshape (int w, int h)
{
    glViewport (0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    glFrustum (-1.0, 1.0, -1.0, 1.0, 1.5, 2*light_position[2]);
    glMatrixMode (GL_MODELVIEW);
//    cerr<<"reshape"<<endl;
}

void keyboard(unsigned char key, int x, int y)
{
    switch (key) {
    case 27:
        exit(0);
        break;
    }
}

void idle()
{
//    cerr<<"animate"<<endl;

    sofaScene.animate();

    glutPostRedisplay();
}

int main(int argc, char** argv)
{
    glutInit(&argc,argv);
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );
    glutInitWindowSize (500, 500);
    glutInitWindowPosition (100, 100);
    glutCreateWindow (argv[0]);
    init();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);

    // --- Parameter initialisation ---
    std::string fileName ;
    std::vector<std::string> plugins;

    sofa::helper::parse()
            .option(&plugins,'l',"load","load given plugins")
            .parameter(&fileName,'f',"file","scene file to load")
            (argc,argv);

    // --- Init sofa ---
    sofaScene.init(plugins,fileName);
    sofaScene.debug = true;

    glutMainLoop();


    return 0;
}
