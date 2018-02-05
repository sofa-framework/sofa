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
/** \example glutOnePick.cpp
 * Basic glut application with picking
 */

/**
  A simple glut application featuring a Sofa simulation.
  Contrary to other projects, this does not use a sofa::gui.

  @author Francois Faure, 2014
  */

#include "oneTetra.h"

#include <iostream>
#include <fstream>
#include <ctime>
#include <GL/glew.h>
#include <sofa/helper/system/glut.h>
using std::endl;
using std::cout;

#include <sofa/helper/ArgumentParser.h>
//#include <SofaSimpleGUI/SofaGLScene.h>
#include <SofaSimpleGUI/SofaScene.h>
#include <SofaSimpleGUI/SofaGL.h>
using namespace sofa::simplegui;

// ---------------------------------------------------------------------
// Sofa interface
// ---------------------------------------------------------------------
//sofa::simplegui::SofaGLScene sofaScene;     ///< The interface of the application with Sofa
sofa::simplegui::SofaScene* sofaScene;     ///< The interface of the application with Sofa
sofa::simplegui::SofaGL* sofaGL;     ///< The interface of the application with the viewer
sofa::simplegui::SpringInteractor* drag = NULL; ///< Mouse interactor

#include <SofaSimpleGUI/Camera.h>
sofa::simplegui::Camera camera;


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
    camera.lookAt(); // apply viewing transform

    sofaGL->draw();

    // display a box, for debug
    glColor3f (1.0, 0.0, 0.0);
    glutWireCube (1.0);
    glutSwapBuffers();

    // Due to some bug, the first call displays nothing. Hence the poor following fix. Contributions are welcome.
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

    sofaScene->step(0.04);
    glutPostRedisplay();
}


void mouseButton(int button, int state, int x, int y)
{
    //    cout<<"mousebutton, modifiers = " << glutGetModifiers() << endl;

    if( interacting && button==GLUT_LEFT_BUTTON && state == GLUT_UP ) // interaction release
    {
        sofaGL->detach(drag);
        delete drag;

        interacting = false;
        return;
    }

    if( isShiftPressed() ){
        switch (button) {
        case GLUT_LEFT_BUTTON:
            if (state == GLUT_DOWN)
            {
                //PickedPoint glpicked = sofaGL->pick(camera_position[0],camera_position[1],camera_position[2], x,y);
                PickedPoint glpicked = sofaGL->pick(camera.eye()[0],camera.eye()[1],camera.eye()[2], x,y);
                if( glpicked )
                {
                    interacting = true;

                    drag = new SpringInteractor(glpicked);
                    sofaGL->attach(drag);
                    //                cout << "Particle glpicked: " << glpicked << endl;
                    sofaScene->printGraph();
                }
                else {
                    cout << "no particle glpicked" << endl;
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
    } else { // shift not pressed
        if( camera.handleMouseButton(
            button==GLUT_LEFT_BUTTON ? Camera::ButtonLeft : button==GLUT_MIDDLE_BUTTON ? Camera::ButtonMiddle : Camera::ButtonRight ,
            state==GLUT_DOWN ? Camera::ButtonDown : Camera::ButtonUp,
            x, y
                    ))
        {
            return;
        }

    }

    glutPostRedisplay();
}

void mouseMotion(int x, int y)
{

    if( interacting )
    {
        sofaGL->move(drag, x,y);
    }
    else{
        if( camera.handleMouseMotion(x,y) )
        {
            glutPostRedisplay();
            return;
        }
    }
}


int main(int argc, char** argv)
{
    glewInit();
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
    std::vector<std::string> plugins;

    sofa::helper::parse("Simple glut application featuring a Sofa scene.")
            .option(&plugins,'l',"load","load given plugins") // example to load Flexible and Compliant: -l Flexible -l Compliant
            .option(&fileName,'f',"file","scene file to load")
            (argc,argv);


    // --- Init sofa ---
    sofaScene = new SofaScene;
    sofaScene->loadPlugins( plugins );
    if( fileName.empty() ){
        cout << "No scene file provided, creating default scene " << endl;
        sofaScene->setScene( oneTetra().get() );
    }
    else
        sofaScene->open(fileName);

    sofaGL = new SofaGL(sofaScene);

    // initial viewpoint
    camera.setlookAt ( 0,0,25,    0.0, 0.0, 0.0,    0.0, 1.0, 0.0);

    // Run main loop
    glutMainLoop();


    return 0;
}

