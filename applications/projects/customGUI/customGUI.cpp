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
#include <sofa/simulation/graph/DAGSimulation.h>
#include <sofa/simulation/tree/TreeSimulation.h>
#include <sofa/core/visual/DrawToolGL.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/component/collision/MouseInteractor.h>
#include <sofa/component/typedef/Sofa_typedef.h>
#include <sofa/simulation/common/InitVisitor.h>
#include <sofa/simulation/common/DeleteVisitor.h>
#include <sofa/component/interactionforcefield/StiffSpringForceField.h>


using namespace sofa;
using simulation::Node;
//typedef sofa::simulation::graph::DAGSimulation ParentSimulation;
typedef sofa::simulation::tree::TreeSimulation ParentSimulation;

/** Prototype used to completely encapsulate the use of Sofa in an OpenGL application, without any standard Sofa GUI.
 *
 * @author Francois Faure, 2014
 * */
class SofaScene : public ParentSimulation
{
    // sofa types should not be exposed
    typedef sofa::defaulttype::Vector3 Vec3;
    typedef sofa::component::container::MechanicalObject< defaulttype::Vec3Types > Vec3DOF;
    typedef sofa::component::collision::RayModel RayCollisionModel;

    Node::SPtr groot; ///< root of the graph
    Node::SPtr sroot; ///< root of the scene, child of groot

    // interaction
    Node::SPtr mouseNode;
    bool interactorInUse;
    Vec3DOF::SPtr      mouseDOF;
    RayCollisionModel::SPtr mouseRay;
    Node::SPtr interactionNode;


public:

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

        groot = sofa::simulation::getSimulation()->createNewGraph("");
        groot->setName("theRoot");


    }

    /**
     * @brief Initialize Sofa and load a scene file
     * @param plugins List of plugins to load
     * @param fileName Scene file to load
     */
    void init( std::vector<std::string>& plugins, const std::string& fileName )
    {

        // --- plugins ---
        for (unsigned int i=0; i<plugins.size(); i++){
            cout<<"SofaScene::init, loading plugin " << plugins[i] << endl;
            sofa::helper::system::PluginManager::getInstance().loadPlugin(plugins[i]);
        }

        sofa::helper::system::PluginManager::getInstance().init();


        // --- Create simulation graph ---
        sroot = sofa::simulation::getSimulation()->load(fileName.c_str());
        if (sroot!=NULL)
        {
            sroot->setName("sceneRoot");
            groot->addChild(sroot);
        }
        else {
            serr << "SofaScene::init, could not load scene " << fileName << sendl;
        }

        ParentSimulation::init(groot.get());

        if( debug ){
            cout<<"SofaScene::init, scene loaded" << endl;
            sofa::simulation::getSimulation()->print(groot.get());
        }

        // --- Interaction
        addMouseNode(groot);
        mouseNode->detachFromGraph();
        interactorInUse = false;
    }

    void printScene()
    {
        sofa::simulation::getSimulation()->print(groot.get());
    }

    void reshape(int,int){}

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

    /** Represents a point picked using the mouse. Todo: put this in the parent class */
    struct PickedPoint: public sofa::component::collision::BodyPicked
    {

        /// distance to a given point
        template <typename P1>
        double distance( P1 p1 ) const {
            double d=0;
            for( int i=0; i<3; i++ )
                d += (p1[i]-point[i])*(p1[i]-point[i]);
            return sqrt(d);
        }

        /// Compute relative distance: near=0, far=1.
        double computeDistance (
                double xnear, double ynear, double znear,
                double xfar, double yfar, double zfar
                ) const
        {
            double d1 = sqrt( (point[0]-xnear)*(point[0]-xnear) + (point[1]-ynear)*(point[1]-ynear) + (point[2]-znear)*(point[2]-znear) );
            double d2 = sqrt( (xfar-xnear)*(xfar-xnear) + (yfar-ynear)*(yfar-ynear) + (zfar-znear)*(zfar-znear) );
            return d1/d2;
        }

    };


    /**
     * @brief rayPick Seeks a particle along the given ray. The direction needs not be normalized.
     * @param ox ray origin x
     * @param oy ray origin y
     * @param oz ray origin z
     * @param dx ray direction x
     * @param dy ray direction y
     * @param dz ray direction z
     * @return the closest point picked along the ray. If no point is picked, the conversion to bool returns false.
     */
    PickedPoint rayPick( double ox, double oy, double oz, double dx, double dy, double dz )
    {
        PickedPoint pickedPoint;
        Vec3 origin(ox,oy,oz), direction(dx,dy,dz);
        direction.normalize();
        double distance = 0.5, distanceGrowth = 0.001; // cone around the ray. todo: set this as class member
        if( debug ){
            cout<< "SofaScene::rayPick from origin " << origin << ", in direction " << direction << endl;
        }
        sofa::simulation::MechanicalPickParticlesVisitor picker(sofa::core::ExecParams::defaultInstance(),
                                                                origin, direction, distance, distanceGrowth );
        picker.execute(groot->getContext());
        if (!picker.particles.empty())
        {
            sofa::core::behavior::BaseMechanicalState *mstate = picker.particles.begin()->second.first;
            pickedPoint.mstate=mstate;
            pickedPoint.indexCollisionElement = picker.particles.begin()->second.second;
            pickedPoint.point[0] = mstate->getPX(pickedPoint.indexCollisionElement);
            pickedPoint.point[1] = mstate->getPY(pickedPoint.indexCollisionElement);
            pickedPoint.point[2] = mstate->getPZ(pickedPoint.indexCollisionElement);
            pickedPoint.dist =  0;
            pickedPoint.rayLength = (pickedPoint.point-origin)*direction;

            if (!interactorInUse)
            {
                interactorInUse = true;
                groot->addChild(mouseNode);
                Vec3DOF::WriteVecCoord xmouse = mouseDOF->writePositions();
                xmouse[0] = pickedPoint.point;
                //                cout<<"mouseDOF position: " << xmouse << endl;

                //                cout<<"SofaScene::raypick, mouse node added " << endl;
                //                sofa::simulation::getSimulation()->print(groot.get());

                // ========  Node with multiple parents to create an interaction using a MultiMapping
                Node* parent = dynamic_cast<Node*>( pickedPoint.mstate->getContext() );
                if( parent ) cout << "picked node " << parent->getName() << endl;
                else cout << "could not get the parent " << endl;
                interactionNode = parent->createChild("InteractionNode");
                cout<<"attaching interaction to " << parent->getName() << endl;
//                mouseNode->addChild(interactionNode);

                // use a spring for interaction
                MechanicalObject3 *mouseDof = dynamic_cast<MechanicalObject3*>(mouseNode->getMechanicalState()); assert(mouseDof);
                MechanicalObject3 *parentDof = dynamic_cast<MechanicalObject3*>(parent->getMechanicalState()); assert(parentDof);
                StiffSpringForceField3::SPtr spring = New<StiffSpringForceField3>(mouseDof,parentDof);
                interactionNode->addObject(spring);
//                spring->getMState1() = mouseNode->getMechanicalState();
//                spring->getMState2() = parent->getMechanicalState();
                spring->addSpring(0,pickedPoint.indexCollisionElement,100,0.1,0.);

//                MechanicalObject3::SPtr mappedDOF = New<MechanicalObject3>(); // to contain particles from the two strings
//                interactionNode->addObject(mappedDOF);

//                SubsetMultiMapping3_to_3::SPtr multimapping = New<SubsetMultiMapping3_to_3>();
//                multimapping->setName("InteractionMultiMapping");
//                multimapping->addInputModel( mouseNode->getMechanicalState() );
//                multimapping->addInputModel( parent->getMechanicalState() );
//                multimapping->addOutputModel( mappedDOF.get() );
//                multimapping->addPoint( mouseNode->getMechanicalState(), 0 );
//                multimapping->addPoint( pickedPoint.mstate, pickedPoint.indexCollisionElement );
//                interactionNode->addObject(multimapping);

//                // Node to handle the extension of the interaction link
//                Node::SPtr extension_node = interactionNode->createChild("InteractionExtensionNode");

//                MechanicalObject1::SPtr extensions = New<MechanicalObject1>();
//                extension_node->addObject(extensions);

//                using component::constraintset::EdgeSetTopologyContainer;
//                EdgeSetTopologyContainer::SPtr edgeSet = New<EdgeSetTopologyContainer>();
//                extension_node->addObject(edgeSet);
//                edgeSet->addEdge(0,1);

//                DistanceMapping31::SPtr extensionMapping = New<DistanceMapping31>();
//                extensionMapping->setModels(mappedDOF.get(),extensions.get());
//                extension_node->addObject( extensionMapping );
//                extensionMapping->setName("InteractionExtension_mapping");


//                UniformCompliance1::SPtr compliance = New<UniformCompliance1>();
//                extension_node->addObject(compliance);
//                compliance->compliance.setName("connectionCompliance");
//                compliance->compliance.setValue(0.0001);


             interactionNode->execute<simulation::InitVisitor>(core::ExecParams::defaultInstance());
             if( debug ){
                 sofa::simulation::getSimulation()->print(groot.get());
                 cout<<"SofaScene::raypick, mouse node added to =============================" << groot->getName() << endl;
             }

            }


        }
        return pickedPoint;
    }

    /// Displace the interaction ray by the given amount
    void rayMove( double dx, double dy, double dz )
    {
//        cout << "SofaScene::rayMove "<< dx <<" "<< dy << " " << dz << endl;
        if( interactorInUse )
        {
            Vec3DOF::WriteVecCoord xmouse = mouseDOF->writePositions();
            xmouse[0] += Vec3(dx,dy,dz);
        }
        else {
            cout<<"SofaScene::rayMove, but interactor not in use. Doing nothing." << endl;
        }
    }

    /// Detach the interaction ray from the scene
    void rayRelease()
    {
        interactionNode->detachFromGraph();
        mouseNode->detachFromGraph();
        interactorInUse = false;
        interactionNode->execute<simulation::DeleteVisitor>(core::ExecParams::defaultInstance());
    }


protected:
    sofa::core::visual::DrawToolGL   drawToolGL;
    sofa::core::visual::VisualParams* vparams;

    void addMouseNode( Node::SPtr groot )
    {
        //        cout<<"ray pick, groot has " << groot->getChildren().size() << " children before" << endl;
        mouseNode = groot->createChild("Mouse");
        //        cout<<"ray pick, groot has " << groot->getChildren().size() << " children after" << endl;

        mouseDOF = sofa::core::objectmodel::New<Vec3DOF>(); mouseDOF->resize(1);
        mouseDOF->setName("MousePosition");
        mouseNode->addObject(mouseDOF);


        mouseRay = sofa::core::objectmodel::New<RayCollisionModel>();
        mouseRay->setNbRay(1);
        mouseRay->getRay(0).setL(1000000);
        mouseRay->setName("MouseCollisionModel");
        mouseNode->addObject(mouseRay);


        mouseNode->init(sofa::core::ExecParams::defaultInstance());
        mouseDOF->init();
        mouseRay->init();
        //                interaction->attach(mouseNode.get());
        interactorInUse=true;

    }


};





// ---------------------------------------------------------------------
// --- A basic glut application featuring a Sofa simulation
// Sofa is invoked only through variable sofaScene.
// ---------------------------------------------------------------------

SofaScene sofaScene; ///< The interface of the application with Sofa
//SofaGlutGui sofaScene; ///< The interface of the application with Sofa
SofaScene::PickedPoint picked; ///< Point picked using the mouse

// Various shared variables for glut
GLfloat light_position[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat light_ambient[] = { 0.0, 0.0, 0.0, 0.0 };
GLfloat light_diffuse[] = { 1.0, 1.0, 1.0, 0.0 };
GLfloat light_specular[] = { 1.0, 1.0, 1.0, 0.0 };

GLfloat camera_position[] = { 0.0, 0.0, 25.0, 0.0 };
GLfloat znear = camera_position[2]-10;
GLfloat zfar = camera_position[2]+10;

bool _isControlPressed = false; bool isControlPressed(){ return _isControlPressed; }
bool _isShiftPressed = false; bool isShiftPressed(){ return _isShiftPressed; }
bool _isAltPressed = false; bool isAltPressed(){ return _isAltPressed; }


GLfloat delta_x = 0.0;
GLfloat delta_y = 0.0;

bool animating = true;
bool interacting = false;



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
    //        glOrtho( -1.0, 1.0, -1.0, 1.0, 10, 2*camera_position[2]);
    //    glFrustum (-1.0, 1.0, -1.0, 1.0, 1.5, 2*camera_position[2]);
    gluPerspective (55.0, (GLfloat) w/(GLfloat) h, znear, zfar );
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


int prev_x, prev_y;


void mouseButton(int button, int state, int x, int y)
{
    //    sofaScene.glut_mouse(button,state,x,y);
    //    return;

    prev_x = x;
    prev_y = y;

    GLint realy;  /*  OpenGL y coordinate position  */

    switch (button) {
    case GLUT_LEFT_BUTTON:
        if (state == GLUT_DOWN)
        {
            GLint viewport[4];
            glGetIntegerv (GL_VIEWPORT, viewport);
            GLdouble mvmatrix[16], projmatrix[16];
            glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
            glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);
            // note viewport[3] is height of window in pixels
            realy = viewport[3] - (GLint) y - 1;
            //cout<< "Coordinates at cursor are ("<<x<<","<<realy<<")"<<endl;
            GLdouble wx, wy, wz;  /*  returned world x, y, z coords  */
            gluUnProject ((GLdouble) x, (GLdouble) realy, 0.0, mvmatrix, projmatrix, viewport, &wx, &wy, &wz); // z=0: near plane
            //cout<<"World coords at z=0.0 are ("<<wx<<","<<wy<<","<<wz<<")"<<endl;
            GLdouble wx1, wy1, wz1;
            gluUnProject ((GLdouble) x, (GLdouble) realy, 1.0, mvmatrix, projmatrix, viewport, &wx1, &wy1, &wz1); // z=1: far plane
            //cout<<"World coords at z=1.0 are ("<< wx <<","<< wy<<","<< wz <<")"<< endl;

            if( (picked = sofaScene.rayPick(camera_position[0],camera_position[1],camera_position[2], wx1-wx, wy1-wy, wz1-wz)) )
            {
                interacting = true;
                picked.dist = picked.computeDistance(wx,wy,wz, wx1,wy1,wz1);
                cout<<"picked particle index " << picked.indexCollisionElement << " in state " << picked.mstate->getName() <<", distance = " << picked.dist <<  ", position = " << picked.point << endl;
            }
            else {
                cout << "no particle picked" << endl;
            }
        }
        else
        {
            sofaScene.rayRelease();
            interacting = false;
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
        GLint viewport[4];
        glGetIntegerv (GL_VIEWPORT, viewport);
        GLdouble mvmatrix[16], projmatrix[16];
        glGetDoublev (GL_MODELVIEW_MATRIX, mvmatrix);
        glGetDoublev (GL_PROJECTION_MATRIX, projmatrix);

        GLdouble p[3], p_prev[3];
        gluUnProject ( x, viewport[3]-y-1, picked.dist, mvmatrix, projmatrix, viewport, &p[0], &p[1], &p[2]); // new position of the picked point
        //        cout<<"x="<< x <<", y="<< y <<", X="<<p[0]<<", Y="<<p[1]<<", Z="<<p[2]<<endl;
        gluUnProject ( prev_x, viewport[3]-prev_y-1, picked.dist, mvmatrix, projmatrix, viewport, &p_prev[0], &p_prev[1], &p_prev[2]); // previous position of the picked point

        sofaScene.rayMove(  p[0]-p_prev[0], p[1]-p_prev[1], p[2]-p_prev[2] );
    }
    else{
    }

    prev_x = x;
    prev_y = y;
}


void update_modifiers()
{
    _isControlPressed =  (glutGetModifiers()&GLUT_ACTIVE_CTRL )!=0;
    _isShiftPressed   =  (glutGetModifiers()&GLUT_ACTIVE_SHIFT)!=0; if( _isShiftPressed ) cout <<"shift pressed" <<endl; else cout<<"shift not pressed" << endl;
    _isAltPressed     =  (glutGetModifiers()&GLUT_ACTIVE_ALT  )!=0;
}
void specialKey(int k, int x, int y)
{
    //    cout<<"special key " << k << endl;
    cout<<"modifiers = " << glutGetModifiers() << endl; // looks like freeglut is currently buggy, since this is always null
    update_modifiers();
    //    sofaScene.glut_special(k,x,y);
}

#include <sofa/config.h>

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

    cout<<"SOFA_BUILD_DIR = " << SOFA_BUILD_DIR << endl;

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

