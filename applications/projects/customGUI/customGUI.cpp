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

/**
  A simple glut application featuring a Sofa simulation.
  Contrary to other projects, this does not use a sofa::gui

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

#include <sofa/helper/system/PluginManager.h>
#include <sofa/component/init.h>
#include <sofa/simulation/common/xml/initXml.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/SetDirectory.h>
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/helper/Factory.h>
#include <sofa/helper/BackTrace.h>
#include <sofa/component/misc/WriteState.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/core/visual/VisualParams.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/collision/MouseInteractor.h>
//#include <sofa/gui/PickHandler.h>

using namespace sofa;
using simulation::Node;
typedef sofa::simulation::graph::DAGSimulation ParentSimulation;
//typedef sofa::simulation::tree::TreeSimulation ParentSimulation;

/** Prototype used to completely encapsulate the use of Sofa in an OpenGL application, without any standard Sofa GUI.
 *
 * @author Francois Faure, 2014
 * */
class SofaScene : public ParentSimulation
{
    // sofa types should not be exposed
    typedef sofa::defaulttype::Vector3 Vec3;
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > MouseContainer;
    typedef sofa::component::collision::RayModel MouseCollisionModel;

    // interaction
    Node::SPtr mouseNode;
    bool interactorInUse;
    MouseContainer::SPtr      mouseContainer;
    MouseCollisionModel::SPtr mouseCollision;


public:
    Node::SPtr groot;
    bool debug;

    SofaScene()
    {
        interactorInUse = false;
        debug = true;


        sofa::simulation::setSimulation(new ParentSimulation());

        sofa::component::init();
        sofa::simulation::xml::initXml();

        vparams = sofa::core::visual::VisualParams::defaultInstance();
        vparams->drawTool() = &drawToolGL;

//        groot = sofa::simulation::getSimulation()->createNewGraph("");



//        mouseNode->detachFromGraph();

    }

    /**
     * @brief Initialize Sofa and load a scene file
     * @param plugins List of plugins to load
     * @param fileName Scene file to load
     */
    void init( std::vector<std::string>& plugins, const std::string& fileName )
    {

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
        groot->setName("theRoot");

        ParentSimulation::init(groot.get());

        if( debug ){
            cout<<"SofaScene::init, scene loaded" << endl;
            sofa::simulation::getSimulation()->print(groot.get());
        }



    }

    /**
     * @brief glDraw Draw the Sofa scene using OpenGL.
     * Requires that an OpenGL context is active.
     */
    void glDraw()
    {
//        if(debug)
//            cout<<"SofaScene::glDraw" << endl;
        sofa::simulation::getSimulation()->draw(vparams,groot.get());
    }

    /**
     * @brief Integrate time by one step and update the Sofa scene.
     */
    void animate()
    {
//        if( debug )
//            cout<<"SofaScene::animate" << endl;
        sofa::simulation::getSimulation()->animate(groot.get(),0.04);
    }

    /**
     * @brief rayPick Seeks a particle along the given ray. The direction needs not be normalized.
     * @param ox ray origin x
     * @param oy ray origin y
     * @param oz ray origin z
     * @param dx ray direction x
     * @param dy ray direction y
     * @param dz ray direction z
     * @return true if a particle is found near the ray
     */
    bool rayPick( double ox, double oy, double oz, double dx, double dy, double dz )
    {
        Vec3 origin(ox,oy,oz), direction(dx,dy,dz);
        direction.normalize();
        double distance = 30, distanceGrowth = 0.001; // cone around the ray
        if( debug ){
            cout<< "SofaScene::rayPick from origin " << origin << ", in direction " << direction << endl;
        }
        sofa::simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(),
                                                                origin, direction, distance, distanceGrowth );
        picker.execute(groot->getContext());
        if (!picker.particles.empty())
        {
            sofa::core::behavior::BaseMechanicalState *mstate = picker.particles.begin()->second.first;
            result.mstate=mstate;
            result.indexCollisionElement = picker.particles.begin()->second.second;
            result.point[0] = mstate->getPX(result.indexCollisionElement);
            result.point[1] = mstate->getPY(result.indexCollisionElement);
            result.point[2] = mstate->getPZ(result.indexCollisionElement);
            result.dist =  0;
            result.rayLength = (result.point-origin)*direction;

            if (!interactorInUse)
            {
                if( debug ){
                    sofa::simulation::getSimulation()->print(groot.get());
                    cout<<"SofaScene::raypick, mouse node to add to =============================" << groot->getName() << endl;
                }
//                groot->addChild(mouseNode);
                // --- Interaction
                cout<<"ray pick, groot has " << groot->getChildren().size() << " children before" << endl;
                mouseNode = groot->createChild("Mouse");
                cout<<"ray pick, groot has " << groot->getChildren().size() << " children after" << endl;

                mouseContainer = sofa::core::objectmodel::New<MouseContainer>(); mouseContainer->resize(1);
                mouseContainer->setName("MousePosition");
                mouseNode->addObject(mouseContainer);


                mouseCollision = sofa::core::objectmodel::New<MouseCollisionModel>();
                mouseCollision->setNbRay(1);
                mouseCollision->getRay(0).setL(1000000);
                mouseCollision->setName("MouseCollisionModel");
                mouseNode->addObject(mouseCollision);


                mouseNode->init(sofa::core::ExecParams::defaultInstance());
                mouseContainer->init();
                mouseCollision->init();
//                interaction->attach(mouseNode.get());
                interactorInUse=true;
                if( debug ){
                    cout<<"SofaScene::raypick, mouse node added" << endl;
                    sofa::simulation::getSimulation()->print(groot.get());
                }
            }


            return true;
        }
        else
            return false;
    }

    void reshape(int,int){}

protected:
    sofa::core::visual::DrawToolGL   drawToolGL;
    sofa::core::visual::VisualParams* vparams;
    sofa::component::collision::BodyPicked result;


};





// ---------------------------------------------------------------------
// --- A basic glut application featuring a Sofa simulation
// Sofa is invoked only through variable sofaScene.
// ---------------------------------------------------------------------

SofaScene sofaScene; ///< The interface of the application with Sofa
//SofaGlutGui sofaScene; ///< The interface of the application with Sofa

// Various shared variables for glut
GLfloat light_position[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat light_specular[] = { 1.0, 1.0, 1.0, 0.0 };

GLfloat spin_x = 0.0;
GLfloat spin_y = 0.0;

bool animating = true;



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
//        glOrtho( -1.0, 1.0, -1.0, 1.0, 10, 2*light_position[2]);
    //    glFrustum (-1.0, 1.0, -1.0, 1.0, 1.5, 2*light_position[2]);
    gluPerspective (55.0, (GLfloat) w/(GLfloat) h, light_position[2]-10, light_position[2]+10 );
    glMatrixMode (GL_MODELVIEW);
    //    cout<<"reshape"<<endl;

    sofaScene.reshape(w,h);
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

//    sofaScene.glut_keyboard(key,x,y);
//    glutPostRedisplay();
}

void idle()
{
    if( !animating )
        return;

    sofaScene.animate();
    glutPostRedisplay();
}


int old_x, old_y;

void mouseButton(int button, int state, int x, int y)
{
//    sofaScene.glut_mouse(button,state,x,y);
//    return;

    old_x = x;
    old_y = y;

    GLint viewport[4];
    GLdouble mvmatrix[16], projmatrix[16];
    GLint realy;  /*  OpenGL y coordinate position  */

    switch (button) {
    case GLUT_LEFT_BUTTON:
        if (state == GLUT_DOWN) {
            glGetIntegerv (GL_VIEWPORT, viewport);
            glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
            glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
            // note viewport[3] is height of window in pixels
            realy = viewport[3] - (GLint) y - 1;
            //cout<< "Coordinates at cursor are ("<<x<<","<<realy<<")"<<endl;
            GLdouble wx, wy, wz;  /*  returned world x, y, z coords  */
            gluUnProject ((GLdouble) x, (GLdouble) realy, 0.0, mvmatrix, projmatrix, viewport, &wx, &wy, &wz); // z=0: near plane
            //cout<<"World coords at z=0.0 are ("<<wx<<","<<wy<<","<<wz<<")"<<endl;
            GLdouble wx1, wy1, wz1;
            gluUnProject ((GLdouble) x, (GLdouble) realy, 1.0, mvmatrix, projmatrix, viewport, &wx1, &wy1, &wz1); // z=0: far plane
            //cout<<"World coords at z=1.0 are ("<< wx <<","<< wy<<","<< wz <<")"<< endl;

            if( sofaScene.rayPick(light_position[0],light_position[1],light_position[2], wx1-wx, wy1-wy, wz1-wz) ){
                cout<<"picked particle " << endl;
            }
            else {
                cout << "no particle picked" << endl;
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
    spin_x = x - old_x;
    spin_y = y - old_y;

//    sofaScene.glut_motion(x,y);
}

bool _isControlPressed = false; bool isControlPressed(){ return _isControlPressed; }
bool _isShiftPressed = false; bool isShiftPressed(){ return _isShiftPressed; }
bool _isAltPressed = false; bool isAltPressed(){ return _isAltPressed; }

void update_modifiers()
{
    _isControlPressed =  (glutGetModifiers()&GLUT_ACTIVE_CTRL )!=0;
    _isShiftPressed   =  (glutGetModifiers()&GLUT_ACTIVE_SHIFT)!=0;
    _isAltPressed     =  (glutGetModifiers()&GLUT_ACTIVE_ALT  )!=0;
}
void specialKey(int k, int x, int y)
{
    update_modifiers();
//    sofaScene.glut_special(k,x,y);
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
    glutMotionFunc(mouseMotion);
    glutMouseFunc(mouseButton);
    glutSpecialFunc( specialKey );

    // --- Parameter initialisation ---
    std::string fileName ;
    std::vector<std::string> plugins;

    sofa::helper::parse("Simple glut application featuring a Sofa scene.")
            .option(&plugins,'l',"load","load given plugins")
            .parameter(&fileName,'f',"file","scene file to load")
            (argc,argv);

    // --- Init sofa ---
    sofaScene.init(plugins,fileName);
//    sofaScene.debug = false;

    glutMainLoop();


    return 0;
}


//#include <sofa/gui/glut/SimpleGUI.h>
//class SofaGlutGui: public sofa::gui::glut::SimpleGUI, public sofa::simulation::graph::DAGSimulation
//{
//public:
//    typedef sofa::gui::glut::SimpleGUI Gui;
//    typedef sofa::simulation::graph::DAGSimulation Simulation;
//    bool debug;
//    SofaGlutGui():debug(true){}
////    ~SofaGlutGui(){}

//    /**
//     * @brief Initialize Sofa and load a scene file
//     * @param plugins List of plugins to load
//     * @param fileName Scene file to load
//     */
//    void init( std::vector<std::string>& plugins, const std::string& fileName )
//    {
//        sofa::simulation::setSimulation(new sofa::simulation::graph::DAGSimulation());

//        sofa::component::init();
//        sofa::simulation::xml::initXml();


//        // --- plugins ---
//        for (unsigned int i=0; i<plugins.size(); i++)
//            sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);

//        sofa::helper::system::PluginManager::getInstance().init();


//        // --- Create simulation graph ---
//        setScene( sofa::simulation::getSimulation()->load(fileName.c_str()) );
//        if (getScene()==NULL)
//        {
//            setScene( sofa::simulation::getSimulation()->createNewGraph("") );
//        }

//        Simulation::init(getScene());

//        if( debug ){
//            cout<<"SofaScene::init, scene loaded" << endl;
//            sofa::simulation::getSimulation()->print(getScene());
//        }

//    }

//    void glDraw() { Simulation::draw(vparams,getScene()); }
//    void animate() { Simulation::animate(getScene()); }
//    void reshape(int,int){}
//protected:
////    sofa::core::visual::VisualParams vparams;
////    sofa::simulation::Node::SPtr groot;
////    sofa::simulation::Node* getScene(){ return groot.get(); }
////    void setScene(sofa::simulation::Node::SPtr g){ groot=g; }

//};
