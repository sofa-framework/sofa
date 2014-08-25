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
/** \example glutOnePick.cpp
 * Basic glut application with picking
 */

/**
  A simple glut application featuring a Sofa simulation.
  Contrary to other projects, this does not use a sofa::gui.

  @author Francois Faure, 2014
  */

#include <iostream>
#include <fstream>
#include <ctime>
#include <GL/glew.h>
#include <GL/glut.h>
using std::endl;
using std::cout;

#include <sofa/helper/ArgumentParser.h>
#include <SofaSimpleGUI/SofaGLScene.h>
using namespace sofa::newgui;

// ---------------------------------------------------------------------
// Sofa interface
// ---------------------------------------------------------------------
sofa::newgui::SofaGLScene sofaScene;     ///< The interface of the application with Sofa
sofa::newgui::SpringInteractor* drag = NULL; ///< Mouse interactor

#include "oneTetra.h"


// ---------------------------------------------------------------------
// Various shared variables for glut
GLfloat light_position[] = { 25.0, 0.0, 25.0, 1.0 }; // w = 0.0 => directional light ; w = 1.0 => point light (hemi) or spot light
GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat light_specular[] = { 1.0, 1.0, 1.0, 0.0 };

GLfloat camera_position[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat znear = camera_position[2]-10;
GLfloat zfar = camera_position[2]+10;

bool isControlPressed(){ return glutGetModifiers()&GLUT_ACTIVE_CTRL; }
bool isShiftPressed(){ return glutGetModifiers()&GLUT_ACTIVE_SHIFT; }
bool isAltPressed(){ return glutGetModifiers()&GLUT_ACTIVE_ALT; }

bool animating = true;
bool interacting = false;



void initGL(void)
{
    glClearColor (0.0, 0.0, 0.0, 0.0);

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);

    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_DEPTH_TEST);
}


void display(void)
{
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity ();

    glLightfv(GL_LIGHT0, GL_POSITION, light_position); // WARNING: positioning light before camera imply that the light will follow the camera
    gluLookAt ( camera_position[0],camera_position[1],camera_position[2], 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);

    sofaScene.glDraw();

    // display a box, for debug
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
    gluPerspective (55.0, (GLfloat) w/(GLfloat) h, znear, zfar );
    glMatrixMode (GL_MODELVIEW);
}

void keyboard(unsigned char key, int /*x*/, int /*y*/)
{

    switch (key) {
    case 27:
        exit(0);
        break;
    }

    if( key == ' ' ){
        animating = !animating;
    }

}

void idle()
{
    if( !animating )
        return;

    sofaScene.animate();
    glutPostRedisplay();
}


void mouseButton(int button, int state, int x, int y)
{
//    cout<<"mousebutton, modifiers = " << glutGetModifiers() << endl;

    switch (button) {
    case GLUT_LEFT_BUTTON:
        if (state == GLUT_DOWN)
        {
            PickedPoint glpicked = sofaScene.pick(camera_position[0],camera_position[1],camera_position[2], x,y);
            if( glpicked )
            {
                interacting = true;

                drag = new SpringInteractor(glpicked);
                sofaScene.attach(drag);
//                cout << "Particle glpicked: " << glpicked << endl;
                sofaScene.printGraph();
            }
            else {
                cout << "no particle glpicked" << endl;
            }
        }
        else
        {       
            if( interacting )
            {
                sofaScene.detach(drag);
                delete drag;

                interacting = false;
            }
        }
        break;
    case GLUT_RIGHT_BUTTON:
        //        if (state == GLUT_DOWN)
        //            exit(0);
        break;
    default:
        break;
    }

    glutPostRedisplay();
}

void mouseMotion(int x, int y)
{

    if( interacting )
    {
        sofaScene.move(drag, x,y);
    }
    else{
    }
}


int main(int argc, char** argv)
{
    glutInit(&argc,argv);
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );
    glutInitWindowSize (500, 500);
    glutInitWindowPosition (100, 100);
    glutCreateWindow (argv[0]);
    initGL();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutIdleFunc(idle);
    glutMotionFunc(mouseMotion);
    glutMouseFunc(mouseButton);

    // --- Parameter initialisation ---
    std::string fileName ;

    sofa::helper::parse("Simple glut application featuring a Sofa scene.")
            .option(&sofaScene.plugins,'l',"load","load given plugins")
            .option(&fileName,'f',"file","scene file to load")
            (argc,argv);

    // --- Init sofa ---
    if( fileName.empty() ){
        cout << "No scene file provided, creating default scene " << endl;
        sofaScene.init( oneTetra() );
    }
    else
        sofaScene.init(fileName);

    glutMainLoop();


    return 0;
}

