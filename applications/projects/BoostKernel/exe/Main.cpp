/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 3      *
*                (c) 2006-2008 MGH, INRIA, USTL, UJF, CNRS                    *
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
* Authors: M. Adam, J. Allard, B. Andre, P-J. Bensoussan, S. Cotin, C. Duriez,*
* H. Delingette, F. Falipou, F. Faure, S. Fonteneau, L. Heigeas, C. Mendoza,  *
* M. Nesme, P. Neumann, J-P. de la Plata Alcade, F. Poyer and F. Roy          *
*                                                                             *
* Contact information: contact@sofa-framework.org                             *
******************************************************************************/
#include <GL/glut.h>
#include "traqueboule.h"
#include <sofa/helper/ArgumentParser.h>
#include "BglModeler.h"
#include "../lib/BglNode.h"
#include <iostream>
using std::cerr;
using std::endl;


typedef sofa::simulation::bgl::BglScene Scene;
typedef sofa::simulation::bgl::BglNode Node;
typedef BglModeler MyModeler;

Scene scene;

bool animating = false;
bool step_by_step = true;


// Actions d'affichage
void display(void);
void display(void)
{
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);

    GLfloat l0_position[] = {0,0,0,1};
    GLfloat l0_ambient[] = {0.1,0.1,0.1};
    GLfloat l0_diffuse[] = {0.8,0.8,0.8};
    GLfloat l0_specular[] = {0.5,0.5,0.5};

    // lumiere 0
    glLightfv( GL_LIGHT0, GL_AMBIENT,   l0_ambient );
    glLightfv( GL_LIGHT0, GL_DIFFUSE,   l0_diffuse );
    glLightfv( GL_LIGHT0, GL_SPECULAR,  l0_specular );


    // Details sur le mode de tracé
    glEnable( GL_DEPTH_TEST );            // effectuer le test de profondeur

    // Effacer tout
    glClearColor (0.0, 0.0, 0.0, 0.0);
    glClear( GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT); // la couleur et le z

    glLoadIdentity();  // repere camera

    glLightfv( GL_LIGHT0, GL_POSITION,  l0_position ); // source liee a l'observateur

    tbVisuTransform(); // origine et orientation de la scene

    scene.glDraw();

    //glColor3f(1,1,1);
    //glutWireCube( 10 );


    glutSwapBuffers();
}

// pour changement de taille ou desiconification
void reshape(int w, int h)
{
    glViewport(0, 0, (GLsizei) w, (GLsizei) h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    //glOrtho (-1.1, 1.1, -1.1,1.1, -1000.0, 1000.0);
    gluPerspective (50, (float)w/h, 1, 100);
    glMatrixMode(GL_MODELVIEW);
}

// prise en compte du clavier
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
    switch (key)
    {
    case 27:     // touche ESC
        exit(0);
    case 32:     // touche SPACE
        animating = !animating;
        break;
    case 's':
        step_by_step = !step_by_step;
        break;
    }
}

void animate();
void animate()
{
    if ( animating )
    {
        //cerr<<"-----one step ------"<<endl;
        scene.animate(0.04);
        glutPostRedisplay();
        if ( step_by_step )
            animating = false;
    }
}


// programme principal
int main(int argc, char** argv)
{


    int W_fen = 600;  // largeur fenetre
    int H_fen = 600;  // hauteur fenetre

    sofa::helper::parse("Basic simulation ")
    .option(&W_fen,'L',"Largeur","largeur de la fenêtre en pixels")
    .option(&H_fen,'H',"Hauteur","hauteur de la fenêtre en pixels")
    (argc,argv);

    tbHelp();                      // affiche l'aide sur la traqueboule
    cout<<endl<<"Press SPACE to animate. Press S to toggle step-by-step mode"<<endl;

    glutInit(&argc, argv);


    MyModeler modeler(&scene);
    /*    modeler.buildOneTetrahedron();
        modeler.buildMixedPendulum();*/
    modeler.buildSceneWithInitDependencies();
    //scene.load("chain2.xml");
    scene.init();
    scene.setShowBehaviorModels(true);
    scene.setShowVisualModels(false);
    scene.setShowCollisionModels(true);
    //scene.setShowMappings(true);
    //scene.setShowMechanicalMappings(true);
    scene.setShowNormals(false);

    // couches du framebuffer utilisees par l'application
    glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH );

    // position et taille de la fenetre
    glutInitWindowPosition(200, 100);
    glutInitWindowSize(W_fen,H_fen);
    glutCreateWindow(argv[0]);


    // Initialisation du point de vue
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0,0,-20);
    tbInitTransform();     // initialisation du point de vue

    // cablage des callback
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyboard);
    glutDisplayFunc(display);
    glutMouseFunc(tbMouseFunc);    // traqueboule utilise la souris
    glutMotionFunc(tbMotionFunc);  // traqueboule utilise la souris
    glutIdleFunc( animate );

    // lancement de la boucle principale
    glutMainLoop();
    return 0;  // instruction jamais exécutée
}

