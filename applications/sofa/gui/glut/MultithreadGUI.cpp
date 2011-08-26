/******************************************************************************
*       SOFA, Simulation Open-Framework Architecture, version 1.0 beta 4      *
*                (c) 2006-2009 MGH, INRIA, USTL, UJF, CNRS                    *
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
#include "MultithreadGUI.h"
#include <sofa/helper/system/config.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/thread/CircularQueue.inl>
#include <sofa/simulation/common/CopyAspectVisitor.h>
#include <sofa/simulation/common/ReleaseAspectVisitor.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/core/ExecParams.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/helper/system/SetDirectory.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>

#include <sofa/helper/gl/glfont.h>
#include <sofa/helper/gl/RAII.h>
#include <sofa/helper/gl/GLSLShader.h>
#include <sofa/helper/io/ImageBMP.h>

#include <sofa/helper/system/thread/CTime.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/LaparoscopicRigidTypes.h>
#include <sofa/defaulttype/BoundingBox.h>

#include <sofa/gui/OperationFactory.h>
#include <sofa/gui/MouseOperations.h>

#ifdef SOFA_HAVE_CHAI3D
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/core/objectmodel/GLInitializedEvent.h>
#endif // SOFA_HAVE_CHAI3D
#ifdef SOFA_SMP
#include <sofa/component/visualmodel/VisualModelImpl.h>
#include <sofa/simulation/common/AnimateBeginEvent.h>
#include <sofa/simulation/common/CollisionVisitor.h>
#include <sofa/simulation/common/AnimateEndEvent.h>
#include <sofa/simulation/common/PropagateEventVisitor.h>
#include <sofa/simulation/common/VisualVisitor.h>
#include <athapascan-1>
#include "Multigraph.inl"
#endif /* SOFA_SMP */
// define this if you want video and OBJ capture to be only done once per N iteration
//#define CAPTURE_PERIOD 5


namespace sofa
{

namespace gui
{

namespace glut
{

using std::cout;
using std::endl;
using namespace sofa::defaulttype;
using namespace sofa::helper::gl;
using sofa::simulation::getSimulation;
#ifdef SOFA_SMP
using namespace sofa::simulation;
struct doCollideTask
{
    void operator()()
    {
        //	std::cout << "Recording simulation with base name: " << writeSceneName << "\n";
        // groot->execute<CollisionVisitor>();
        // TODO MultithreadGUI::instance->getScene()->execute<CollisionVisitor>();
        // TODO AnimateBeginEvent ev ( 0.0 );
        // TODO PropagateEventVisitor act ( &ev );
        // TODO MultithreadGUI::instance->getScene()->execute ( act );
        //	sofa::simulation::tree::getSimulation()->animate(groot);

    }
};
struct animateTask
{
    void operator()()
    {
        //	std::cout << "Recording simulation with base name: " << writeSceneName << "\n";

        getSimulation()->animate( MultithreadGUI::instance->getScene());

    }
};

struct collideTask
{
    void operator()()
    {
        //	std::cout << "Recording simulation with base name: " << writeSceneName << "\n";

        //   a1::Fork<doCollideTask>()();
        //	sofa::simulation::tree::getSimulation()->animate(groot);

    }
};
struct visuTask
{
    void operator()()
    {
        //	std::cout << "Recording simulation with base name: " << writeSceneName << "\n";
        // TODO AnimateEndEvent ev ( 0.0 );
        // TODO PropagateEventVisitor act ( &ev );
        // TODO MultithreadGUI::instance->getScene()->execute ( act );
        // TODO MultithreadGUI::instance->getScene()->execute<VisualUpdateVisitor>();

    }
};
struct MainLoopTask
{

    void operator()()
    {
        //	std::cout << "Recording simulation with base name: " << writeSceneName << "\n";
        Iterative::Fork<doCollideTask>()();
        Iterative::Fork<animateTask >(a1::SetStaticSched(1,1,Sched::PartitionTask::SUBGRAPH))();
        Iterative::Fork<visuTask>()();
        //a1::Fork<collideTask>(a1::SetStaticSched(1,1,Sched::PartitionTask::SUBGRAPH))();
    }
};
#endif /* SOFA_SMP */

MultithreadGUI* MultithreadGUI::instance = NULL;

// ---------------------------------------------------------
// --- Multithread related stuff
// ---------------------------------------------------------

void MultithreadGUI::initAspects()
{
    aspectPool.setReleaseCallback(boost::bind(&MultithreadGUI::releaseAspect, this, _1));
    simuAspect = aspectPool.allocate();
    core::ExecParams::defaultInstance()->setAspectID(simuAspect->aspectID());
}

void MultithreadGUI::initThreads()
{
    closeSimu = false;
    simuThread.reset(new boost::thread(boost::bind(&MultithreadGUI::simulationLoop, this)));
}

void MultithreadGUI::closeThreads()
{
    closeSimu = true;
    simuThread->join();
}

void MultithreadGUI::simulationLoop()
{
    core::ExecParams* ep = core::ExecParams::defaultInstance();
    ep->setAspectID(simuAspect->aspectID());
    groot->getContext()->setAnimate(true);
    while(!closeSimu)
    {
        step();
        //boost::thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(500));
        AspectRef renderAspect = aspectPool.allocate();
        //fprintf(stderr, "Allocated aspect %d for render\nCopy from %d to %d\n", renderAspect->aspectID(), simuAspect->aspectID(), renderAspect->aspectID());
        simulation::CopyAspectVisitor copyAspect(ep, renderAspect->aspectID(), simuAspect->aspectID());
        groot->execute(copyAspect);
        renderMsgQueue.push(renderAspect);
    }
}

void MultithreadGUI::processMessages()
{
    fprintf(stderr, "Render begin\n");
    do
    {
        while(!renderMsgQueue.isEmpty())
        {
            renderMsgQueue.pop(glAspect);
            fprintf(stderr, "pop aspect\n");
        }
        if(glAspect == 0)
        {
            boost::thread::sleep(boost::get_system_time() + boost::posix_time::milliseconds(20));
            fprintf(stderr, "null aspect\n");
            continue;
        }

    }
    while(glAspect == 0);

    core::ExecParams::defaultInstance()->setAspectID(glAspect->aspectID());
//    core::ExecParams::defaultInstance()->setAspectID(0);
    fprintf(stderr, "Using aspect %d for display\n", core::ExecParams::defaultInstance()->aspectID());
}

void MultithreadGUI::releaseAspect(int aspect)
{
    simulation::ReleaseAspectVisitor releaseAspect(core::ExecParams::defaultInstance(), aspect);
    groot->execute(releaseAspect);
}

// ---------------------------------------------------------
// --- End of Multithread related stuff
// ---------------------------------------------------------

int MultithreadGUI::mainLoop()
{
#ifdef SOFA_SMP
    if(groot)
    {
// TODO	getScene()->execute<CollisionVisitor>();
        a1::Sync();
        mg=new Iterative::Multigraph<MainLoopTask>();
        mg->compile();
        mg->deploy();

    }
#endif /* SOFA_SMP */
    instance->initThreads();

    glutMainLoop();
    return 0;
}

void MultithreadGUI::redraw()
{
    glutPostRedisplay();
}

int MultithreadGUI::closeGUI()
{
    delete this;
    return 0;
}


SOFA_DECL_CLASS(MultithreadGUI)

int MultithreadGUI::InitGUI(const char* /*name*/, const std::vector<std::string>& /*options*/)
{
    return 0;
}

SofaGUI* MultithreadGUI::CreateGUI(const char* /*name*/, const std::vector<std::string>& /*options*/, sofa::simulation::Node* groot, const char* filename)
{

    glutInitDisplayMode ( GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE );

    //glutInitWindowPosition ( x0, y0 );
    //glutInitWindowSize ( nx, ny );
    glutCreateWindow ( ":: SOFA ::" );



    std::cout << "Window created:"
            << " red="<<glutGet(GLUT_WINDOW_RED_SIZE)
            << " green="<<glutGet(GLUT_WINDOW_GREEN_SIZE)
            << " blue="<<glutGet(GLUT_WINDOW_BLUE_SIZE)
            << " alpha="<<glutGet(GLUT_WINDOW_ALPHA_SIZE)
            << " depth="<<glutGet(GLUT_WINDOW_DEPTH_SIZE)
            << " stencil="<<glutGet(GLUT_WINDOW_STENCIL_SIZE)
            << std::endl;

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

    MultithreadGUI* gui = new MultithreadGUI();
    gui->initAspects();
    gui->setScene(groot, filename);

    gui->initializeGL();

#ifdef SOFA_HAVE_CHAI3D
    // Tell nodes that openGl is initialized
    // especialy the GL_MODELVIEW_MATRIX
    sofa::core::objectmodel::GLInitializedEvent ev;
    sofa::simulation::PropagateEventVisitor act(&ev);
    groot->execute(act);
#endif // SOFA_HAVE_CHAI3D

    return gui;
}



void MultithreadGUI::glut_display()
{
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    if (instance)
    {
        instance->processMessages();
        instance->paintGL();
    }
    glutSwapBuffers ();
}

void MultithreadGUI::glut_reshape(int w, int h)
{
    if (instance)
    {
        instance->resizeGL(w,h);
    }
}

void MultithreadGUI::glut_keyboard(unsigned char k, int, int)
{
    if (instance)
    {
        instance->updateModifiers();
        instance->keyPressEvent(k);
    }
}

void MultithreadGUI::glut_mouse(int button, int state, int x, int y)
{
    if (instance)
    {
        instance->updateModifiers();
        instance->mouseEvent( (state==GLUT_DOWN?MouseButtonPress : MouseButtonRelease), x, y, button );
    }
}

void MultithreadGUI::glut_motion(int x, int y)
{
    if (instance)
    {
        //instance->updateModifiers();
        instance->mouseEvent( MouseMove, x, y, 0 );
    }
}

void MultithreadGUI::glut_special(int k, int, int)
{
    if (instance)
    {
        instance->updateModifiers();
        instance->keyPressEvent(k);
    }
}

void MultithreadGUI::glut_idle()
{
//    if (instance)
//    {
//        if (instance->getScene() && instance->getScene()->getContext()->getAnimate())
//            instance->step();
//        else
//            CTime::sleep(0.01);
//        instance->animate();
//    }
    if (instance)
    {
        CTime::sleep(0.001);
        instance->animate();
    }
}






#ifdef SOFA_HAVE_GLEW

// Shadow Mapping parameters

// These store our width and height for the shadow texture
enum { SHADOW_WIDTH = 512 };
enum { SHADOW_HEIGHT = 512 };
enum { SHADOW_MASK_SIZE = 2048 };

// This is used to set the mode with glTexParameteri() for comparing depth values.
// We use GL_COMPARE_R_TO_TEXTURE_ARB as our mode.  R is used to represent the depth value.
#define GL_TEXTURE_COMPARE_MODE_ARB       0x884C

// This is used to set the function with glTexParameteri() to tell OpenGL how we
// will compare the depth values (we use GL_LEQUAL (less than or equal)).
#define GL_TEXTURE_COMPARE_FUNC_ARB       0x884D

// This mode is what will compare our depth values for shadow mapping
#define GL_COMPARE_R_TO_TEXTURE_ARB       0x884E

// The texture array where we store our image data
static GLuint g_DepthTexture;

// This is our global shader object that will load the shader files
static GLSLShader g_Shader;

//float g_DepthOffset[2] = { 3.0f, 0.0f };
static float g_DepthOffset[2] = { 10.0f, 0.0f };
static float g_DepthBias[2] = { 0.0f, 0.0f };

// These are the light's matrices that need to be stored
static float g_mProjection[16] = {0};
static float g_mModelView[16] = {0};
//float g_mCameraInverse[16] = {0};

static GLuint ShadowTextureMask;

// End of Shadow Mapping Parameters
#endif // SOFA_HAVE_GLEW

// ---------------------------------------------------------
// --- Constructor
// ---------------------------------------------------------
MultithreadGUI::MultithreadGUI()
{
    instance = this;

    groot = NULL;
    initTexturesDone = false;
    // setup OpenGL mode for the window

    _previousEyePos = Vector3(0.0, 0.0, 0.0);
    _zoom = 1.0;
    _zoomSpeed = 250.0;
    _panSpeed = 25.0;
    _navigationMode = TRACKBALL_MODE;
    _spinning = false;
    _moving = false;
    _video = false;
    _animationOBJ = false;
    _axis = false;
    _background = 0;
    _shadow = false;
    _numOBJmodels = 0;
    _materialMode = 0;
    _facetNormal = GL_FALSE;
    _renderingMode = GL_RENDER;
    _mouseTrans = false;
    _mouseRotate = false;
    sceneBBoxIsValid = false;
    texLogo = NULL;

    /*_surfaceModel = NULL;
    _springMassView = NULL;
    _mapView = NULL;
    sphViewer = NULL;
    */
    _arrow = gluNewQuadric();
    gluQuadricDrawStyle(_arrow, GLU_FILL);
    gluQuadricOrientation(_arrow, GLU_OUTSIDE);
    gluQuadricNormals(_arrow, GLU_SMOOTH);

    _tube = gluNewQuadric();
    gluQuadricDrawStyle(_tube, GLU_FILL);
    gluQuadricOrientation(_tube, GLU_OUTSIDE);
    gluQuadricNormals(_tube, GLU_SMOOTH);

    _sphere = gluNewQuadric();
    gluQuadricDrawStyle(_sphere, GLU_FILL);
    gluQuadricOrientation(_sphere, GLU_OUTSIDE);
    gluQuadricNormals(_sphere, GLU_SMOOTH);

    _disk = gluNewQuadric();
    gluQuadricDrawStyle(_disk, GLU_FILL);
    gluQuadricOrientation(_disk, GLU_OUTSIDE);
    gluQuadricNormals(_disk, GLU_SMOOTH);

    // init trackball rotation matrix / quaternion
    _newTrackball.ComputeQuaternion(0.0, 0.0, 0.0, 0.0);
    _newQuat = _newTrackball.GetQuaternion();

    ////////////////
    // Interactor //
    ////////////////
    _mouseInteractorMoving = false;
    _mouseInteractorTranslationMode = false;
    _mouseInteractorRotationMode = false;
    _mouseInteractorSavedPosX = 0;
    _mouseInteractorSavedPosY = 0;
    _mouseInteractorTrackball.ComputeQuaternion(0.0, 0.0, 0.0, 0.0);
    _mouseInteractorNewQuat = _mouseInteractorTrackball.GetQuaternion();

    //////////////////////
    m_isControlPressed = false;
    m_isShiftPressed = false;
    m_isAltPressed = false;
    m_dumpState = false;
    m_dumpStateStream = 0;
    m_displayComputationTime = false;
    m_exportGnuplot = false;

    //Register the different Operations possible
    RegisterOperation("Attach").add< AttachOperation >();
    RegisterOperation("Fix").add< FixOperation >();
    RegisterOperation("Incise").add< InciseOperation >();
    RegisterOperation("Remove").add< TopologyOperation >();

    //Add to each button of the mouse an operation
    pick.changeOperation(LEFT,   "Attach");
    pick.changeOperation(MIDDLE, "Incise");
    pick.changeOperation(RIGHT,  "Remove");

    vparams.drawTool() = &drawTool;
}


// ---------------------------------------------------------
// --- Destructor
// ---------------------------------------------------------
MultithreadGUI::~MultithreadGUI()
{
    closeThreads();
    if (instance == this) instance = NULL;
}

// -----------------------------------------------------------------
// --- OpenGL initialization method - includes light definitions,
// --- color tracking, etc.
// -----------------------------------------------------------------
void MultithreadGUI::initializeGL(void)
{
    static GLfloat    specref[4];
    static GLfloat    ambientLight[4];
    static GLfloat    diffuseLight[4];
    static GLfloat    specular[4];
    static GLfloat    lmodel_ambient[]    = {0.0f, 0.0f, 0.0f, 0.0f};
    static GLfloat    lmodel_twoside[]    = {GL_FALSE};
    static GLfloat    lmodel_local[]        = {GL_FALSE};
    static bool        initialized            = false;

    if (!initialized)
    {
        // Define light parameters
        //_lightPosition[0] = 0.0f;
        //_lightPosition[1] = 10.0f;
        //_lightPosition[2] = 0.0f;
        //_lightPosition[3] = 1.0f;

        _lightPosition[0] = -0.7f;
        _lightPosition[1] = 0.3f;
        _lightPosition[2] = 0.0f;
        _lightPosition[3] = 1.0f;

        ambientLight[0] = 0.5f;
        ambientLight[1] = 0.5f;
        ambientLight[2] = 0.5f;
        ambientLight[3] = 1.0f;

        diffuseLight[0] = 0.9f;
        diffuseLight[1] = 0.9f;
        diffuseLight[2] = 0.9f;
        diffuseLight[3] = 1.0f;

        specular[0] = 1.0f;
        specular[1] = 1.0f;
        specular[2] = 1.0f;
        specular[3] = 1.0f;

        specref[0] = 1.0f;
        specref[1] = 1.0f;
        specref[2] = 1.0f;
        specref[3] = 1.0f;
        // Here we initialize our multi-texturing functions
#ifdef SOFA_HAVE_GLEW
        glewInit();
        if (!GLEW_ARB_multitexture)
            std::cerr << "Error: GL_ARB_multitexture not supported\n";
#endif

        _clearBuffer = GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT;
        _lightModelTwoSides = false;

        glDepthFunc(GL_LEQUAL);
        glClearDepth(1.0);
        glEnable(GL_NORMALIZE);

        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

        // Set light model
        glLightModelfv(GL_LIGHT_MODEL_LOCAL_VIEWER, lmodel_local);
        glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);

        // Setup 'light 0'
        glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight);
        glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
        glLightfv(GL_LIGHT0, GL_POSITION, _lightPosition);

        // Enable color tracking
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

        // All materials hereafter have full specular reflectivity with a high shine
        glMaterialfv(GL_FRONT, GL_SPECULAR, specref);
        glMateriali(GL_FRONT, GL_SHININESS, 128);

        glShadeModel(GL_SMOOTH);

        // Define background color
        glClearColor(0.0589f, 0.0589f, 0.0589f, 1.0f);

        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        //Load texture for logo
        texLogo = new helper::gl::Texture(new helper::io::ImageBMP( sofa::helper::system::DataRepository.getFile("textures/SOFA_logo.bmp")));
        texLogo->init();

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);

        // Turn on our light and enable color along with the light
        //glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        //glEnable(GL_COLOR_MATERIAL);

#ifdef SOFA_HAVE_GLEW
        // Here we allocate memory for our depth texture that will store our light's view
        CreateRenderTexture(g_DepthTexture, SHADOW_WIDTH, SHADOW_HEIGHT, GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT);
        CreateRenderTexture(ShadowTextureMask, SHADOW_MASK_SIZE, SHADOW_MASK_SIZE, GL_LUMINANCE, GL_LUMINANCE);

        if (_glshadow == GLSLShader::InitGLSL())
        {
            // Here we pass in our new vertex and fragment shader files to our shader object.
            g_Shader.InitShaders(sofa::helper::system::DataRepository.getFile("shaders/ShadowMappingPCF.vert"), sofa::helper::system::DataRepository.getFile("shaders/ShadowMappingPCF.frag"));
        }
        else
#endif
        {
            printf("WARNING MultithreadGUI : shadows are not supported !\n");
            _shadow = false;
        }

        // change status so we only do this stuff once
        initialized = true;

        _beginTime = CTime::getTime();

        printf("\n");
    }

    // switch to preset view
    SwitchToPresetView();
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------

///////////////////////////////// STORE LIGHT MATRICES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////    This function positions our view from the light for shadow mapping
/////
///////////////////////////////// STORE LIGHT MATRICES \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

void MultithreadGUI::StoreLightMatrices()
{
    //    _lightPosition[0] =  _sceneTransform.translation[0] + 10;//*cosf(TT);
    //    _lightPosition[1] =  _sceneTransform.translation[1] + 10;//*sinf(2*TT);
    //    _lightPosition[2] =  _sceneTransform.translation[2] + 35;//

    //_lightPosition[0] =  1;
    //_lightPosition[1] =  -10;
    //_lightPosition[2] =  0;

    // Reset our current light matrices
    memset(g_mModelView, 0, sizeof(float)*16);
    memset(g_mProjection, 0, sizeof(float)*16);

    g_mModelView[0] = 1; // identity
    g_mModelView[5] = 1;
    g_mModelView[10] = 1;
    g_mModelView[15] = 1;

    // Using perpective shadow map for the "miner lamp" case ( i.e. light.z == 0 )
    // which is just a "rotation" in sceen space

    float lx = -_lightPosition[0] * lastProjectionMatrix[0] - _lightPosition[1] * lastProjectionMatrix[4] + lastProjectionMatrix[12];
    float ly = -_lightPosition[0] * lastProjectionMatrix[1] - _lightPosition[1] * lastProjectionMatrix[5] + lastProjectionMatrix[13];
    float lz = -_lightPosition[0] * lastProjectionMatrix[2] - _lightPosition[1] * lastProjectionMatrix[6] + lastProjectionMatrix[14];
    //float lw = -_lightPosition[0] * lastProjectionMatrix[3] - _lightPosition[1] * lastProjectionMatrix[7] + lastProjectionMatrix[15];
    //std::cout << "lx = "<<lx<<" ly = "<<ly<<" lz = "<<lz<<" lw = "<<lw<<std::endl;

    Vector3 l(-lx,-ly,-lz);
    Vector3 y;
    y = l.cross(Vector3(1,0,0));
    Vector3 x;
    x = y.cross(l);
    l.normalize();
    y.normalize();
    x.normalize();

    g_mProjection[ 0] = x[0]; g_mProjection[ 4] = x[1]; g_mProjection[ 8] = x[2]; g_mProjection[12] =    0;
    g_mProjection[ 1] = y[0]; g_mProjection[ 5] = y[1]; g_mProjection[ 9] = y[2]; g_mProjection[13] =    0;
    g_mProjection[ 2] = l[0]; g_mProjection[ 6] = l[1]; g_mProjection[10] = l[2]; g_mProjection[14] =    0;
    g_mProjection[ 3] =    0; g_mProjection[ 7] =    0; g_mProjection[11] =    0; g_mProjection[15] =    1;

    g_mProjection[ 0] = x[0]; g_mProjection[ 4] = y[0]; g_mProjection[ 8] = l[0]; g_mProjection[12] =    0;
    g_mProjection[ 1] = x[1]; g_mProjection[ 5] = y[1]; g_mProjection[ 9] = l[1]; g_mProjection[13] =    0;
    g_mProjection[ 2] = x[2]; g_mProjection[ 6] = y[2]; g_mProjection[10] = l[2]; g_mProjection[14] =    0;
    g_mProjection[ 3] =    0; g_mProjection[ 7] =    0; g_mProjection[11] =    0; g_mProjection[15] =    1;

    glPushMatrix();
    {

        glLoadIdentity();
        glScaled(1.0/(fabs(g_mProjection[0])+fabs(g_mProjection[4])+fabs(g_mProjection[8])),
                1.0/(fabs(g_mProjection[1])+fabs(g_mProjection[5])+fabs(g_mProjection[9])),
                1.0/(fabs(g_mProjection[2])+fabs(g_mProjection[6])+fabs(g_mProjection[10])));
        glMultMatrixf(g_mProjection);
        glMultMatrixd(lastProjectionMatrix);

        // Grab the current matrix that will be used for the light's projection matrix
        glGetFloatv(GL_MODELVIEW_MATRIX, g_mProjection);

        // Go back to the original matrix
    } glPopMatrix();

    /*
    // Let's push on a new matrix so we don't change the rest of the world
    glPushMatrix();{

        // Reset the current modelview matrix
        glLoadIdentity();

        // This is where we set the light's position and view.
        gluLookAt(_lightPosition[0],  _lightPosition[1],  _lightPosition[2],
        _sceneTransform.translation[0],       _sceneTransform.translation[1],        _sceneTransform.translation[2],        0, 1, 0);

        // Now that we have the light's view, let's save the current modelview matrix.
        glGetFloatv(GL_MODELVIEW_MATRIX, g_mModelView);

        // Reset the current matrix
        glLoadIdentity();

        // Set our FOV, aspect ratio, then near and far planes for the light's view
        gluPerspective(90.0f, 1.0f, 4.0f, 250.0f);

        // Grab the current matrix that will be used for the light's projection matrix
        glGetFloatv(GL_MODELVIEW_MATRIX, g_mProjection);

        // Go back to the original matrix
    }glPopMatrix();
    */
}

/////////////////////////////// CREATE RENDER TEXTURE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////    This function creates a blank texture to render to
/////
/////////////////////////////// CREATE RENDER TEXTURE \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

void MultithreadGUI::CreateRenderTexture(GLuint& textureID, int sizeX, int sizeY, int channels, int type)
{
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Create the texture and store it on the video card
    glTexImage2D(GL_TEXTURE_2D, 0, channels, sizeX, sizeY, 0, type, GL_UNSIGNED_INT, NULL);

    // Set the texture quality
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
}

//////////////////////////////// APPLY SHADOW MAP \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*
/////
/////    This function applies the shadow map to our world data
/////
//////////////////////////////// APPLY SHADOW MAP \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\*

void MultithreadGUI::ApplyShadowMap()
{
#ifdef SOFA_HAVE_GLEW
    // Let's turn our shaders on for doing shadow mapping on our world
    g_Shader.TurnOn();

    // Turn on our texture unit for shadow mapping and bind our depth texture
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, g_DepthTexture);

    // Give GLSL our texture unit that holds the shadow map
    g_Shader.SetInt(g_Shader.GetVariable("shadowMap"), 1);
    //g_Shader.SetInt(g_Shader.GetVariable("tex"), 0);

    // Here is where we set the mode and function for shadow mapping with shadow2DProj().

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE_ARB,
            GL_COMPARE_R_TO_TEXTURE_ARB);

    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC_ARB, GL_LEQUAL);

    // Create our bias matrix to have a 0 to 1 ratio after clip space
    const float mBias[] = {0.5, 0.0, 0.0, 0.0,
            0.0, 0.5, 0.0, 0.0,
            0.0, 0.0, 0.5+g_DepthBias[0], 0.0,
            0.5, 0.5, 0.5+g_DepthBias[1], 1.0
                          };

    glMatrixMode(GL_TEXTURE);

    glLoadMatrixf(mBias);            // The bias matrix to convert to a 0 to 1 ratio
    glMultMatrixf(g_mProjection);    // The light's projection matrix
    glMultMatrixf(g_mModelView);    // The light's modelview matrix
    //glMultMatrixf(g_mCameraInverse);// The inverse modelview matrix

    glMatrixMode(GL_MODELVIEW);            // Switch back to normal modelview mode

    glActiveTextureARB(GL_TEXTURE0_ARB);

    // Render the world that needs to be shadowed

    glPushMatrix();
    {
        glLoadIdentity();
        _sceneTransform.Apply();
        glGetDoublev(GL_MODELVIEW_MATRIX,lastModelviewMatrix);
        DisplayOBJs();
    }
    glPopMatrix();

    // Reset the texture matrix
    glMatrixMode(GL_TEXTURE);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);

    // Turn the first multi-texture pass off

    glActiveTextureARB(GL_TEXTURE1_ARB);
    glDisable(GL_TEXTURE_2D);
    glActiveTextureARB(GL_TEXTURE0_ARB);

    // Light expected, we need to turn our shader off since we are done
    g_Shader.TurnOff();
#endif
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void MultithreadGUI::PrintString(void* font, char* string)
{
    int    len, i;

    len = (int) strlen(string);
    for (i = 0; i < len; i++)
    {
        glutBitmapCharacter(font, string[i]);
    }
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void MultithreadGUI::Display3DText(float x, float y, float z, char* string)
{
    char*    c;

    glPushMatrix();
    glTranslatef(x, y, z);
    for (c = string; *c != '\0'; c++)
    {
        glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
    }
    glPopMatrix();
}

// ---------------------------------------------------
// ---
// ---
// ---------------------------------------------------
void MultithreadGUI::DrawAxis(double xpos, double ypos, double zpos,
        double arrowSize)
{
    float    fontScale    = (float) (arrowSize / 600.0);

    Enable<GL_DEPTH_TEST> depth;
    Enable<GL_LIGHTING> lighting;
    Enable<GL_COLOR_MATERIAL> colorMat;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(GL_SMOOTH);

    // --- Draw the "X" axis in red
    glPushMatrix();
    glColor3f(1.0, 0.0, 0.0);
    glTranslated(xpos, ypos, zpos);
    glRotatef(90.0f, 0.0, 1.0, 0.0);
    gluCylinder(_tube, arrowSize / 50.0, arrowSize / 50.0, arrowSize, 10, 10);
    glTranslated(0.0, 0.0, arrowSize);
    gluCylinder(_arrow, arrowSize / 15.0, 0.0, arrowSize / 5.0, 10, 10);
    // ---- Display a "X" near the tip of the arrow
    glTranslated(-0.5 * fontScale * (double)
            glutStrokeWidth(GLUT_STROKE_ROMAN, 88),
            arrowSize / 15.0, arrowSize /
            5.0);
    glLineWidth(3.0);
    glScalef(fontScale, fontScale, fontScale);
    glutStrokeCharacter(GLUT_STROKE_ROMAN, 88);
    glScalef(1.0f / fontScale, 1.0f / fontScale, 1.0f / fontScale);
    glLineWidth(1.0f);
    // --- Undo transforms
    glTranslated(-xpos, -ypos, -zpos);
    glPopMatrix();

    // --- Draw the "Y" axis in green
    glPushMatrix();
    glColor3f(0.0, 1.0, 0.0);
    glTranslated(xpos, ypos, zpos);
    glRotatef(-90.0f, 1.0, 0.0, 0.0);
    gluCylinder(_tube, arrowSize / 50.0, arrowSize / 50.0, arrowSize, 10, 10);
    glTranslated(0.0, 0.0, arrowSize);
    gluCylinder(_arrow, arrowSize / 15.0, 0.0, arrowSize / 5.0, 10, 10);
    // ---- Display a "Y" near the tip of the arrow
    glTranslated(-0.5 * fontScale * (double)
            glutStrokeWidth(GLUT_STROKE_ROMAN, 89),
            arrowSize / 15.0, arrowSize /
            5.0);
    glLineWidth(3.0);
    glScalef(fontScale, fontScale, fontScale);
    glutStrokeCharacter(GLUT_STROKE_ROMAN, 89);
    glScalef(1.0f / fontScale, 1.0f / fontScale, 1.0f / fontScale);
    glLineWidth(1.0);
    // --- Undo transforms
    glTranslated(-xpos, -ypos, -zpos);
    glPopMatrix();

    // --- Draw the "Z" axis in blue
    glPushMatrix();
    glColor3f(0.0, 0.0, 1.0);
    glTranslated(xpos, ypos, zpos);
    glRotatef(0.0f, 1.0, 0.0, 0.0);
    gluCylinder(_tube, arrowSize / 50.0, arrowSize / 50.0, arrowSize, 10, 10);
    glTranslated(0.0, 0.0, arrowSize);
    gluCylinder(_arrow, arrowSize / 15.0, 0.0, arrowSize / 5.0, 10, 10);
    // ---- Display a "Z" near the tip of the arrow
    glTranslated(-0.5 * fontScale * (double)
            glutStrokeWidth(GLUT_STROKE_ROMAN, 90),
            arrowSize / 15.0, arrowSize /
            5.0);
    glLineWidth(3.0);
    glScalef(fontScale, fontScale, fontScale);
    glutStrokeCharacter(GLUT_STROKE_ROMAN, 90);
    glScalef(1.0f / fontScale, 1.0f / fontScale, 1.0f / fontScale);
    glLineWidth(1.0);
    // --- Undo transforms
    glTranslated(-xpos, -ypos, -zpos);
    glPopMatrix();
}

// ---------------------------------------------------
// ---
// ---
// ---------------------------------------------------
void MultithreadGUI::DrawBox(double* minBBox, double* maxBBox, double r)
{
    //std::cout << "box = < " << minBBox[0] << ' ' << minBBox[1] << ' ' << minBBox[2] << " >-< " << maxBBox[0] << ' ' << maxBBox[1] << ' ' << maxBBox[2] << " >"<< std::endl;
    if (r==0.0)
        r = (Vector3(maxBBox) - Vector3(minBBox)).norm() / 500;
#if 0
    {
        Enable<GL_DEPTH_TEST> depth;
        Disable<GL_LIGHTING> lighting;
        glColor3f(0.0, 1.0, 1.0);
        glBegin(GL_LINES);
        for (int corner=0; corner<4; ++corner)
        {
            glVertex3d(           minBBox[0]           ,
                    (corner&1)?minBBox[1]:maxBBox[1],
                    (corner&2)?minBBox[2]:maxBBox[2]);
            glVertex3d(           maxBBox[0]           ,
                    (corner&1)?minBBox[1]:maxBBox[1],
                    (corner&2)?minBBox[2]:maxBBox[2]);
        }
        for (int corner=0; corner<4; ++corner)
        {
            glVertex3d((corner&1)?minBBox[0]:maxBBox[0],
                    minBBox[1]           ,
                    (corner&2)?minBBox[2]:maxBBox[2]);
            glVertex3d((corner&1)?minBBox[0]:maxBBox[0],
                    maxBBox[1]           ,
                    (corner&2)?minBBox[2]:maxBBox[2]);
        }

        // --- Draw the Z edges
        for (int corner=0; corner<4; ++corner)
        {
            glVertex3d((corner&1)?minBBox[0]:maxBBox[0],
                    (corner&2)?minBBox[1]:maxBBox[1],
                    minBBox[2]           );
            glVertex3d((corner&1)?minBBox[0]:maxBBox[0],
                    (corner&2)?minBBox[1]:maxBBox[1],
                    maxBBox[2]           );
        }
        glEnd();
        return;
    }
#endif
    Enable<GL_DEPTH_TEST> depth;
    Enable<GL_LIGHTING> lighting;
    Enable<GL_COLOR_MATERIAL> colorMat;

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glShadeModel(GL_SMOOTH);

    // --- Draw the corners
    glColor3f(0.0, 1.0, 1.0);
    for (int corner=0; corner<8; ++corner)
    {
        glPushMatrix();
        glTranslated((corner&1)?minBBox[0]:maxBBox[0],
                (corner&2)?minBBox[1]:maxBBox[1],
                (corner&4)?minBBox[2]:maxBBox[2]);
        gluSphere(_sphere,2*r,20,10);
        glPopMatrix();
    }

    glColor3f(1.0, 1.0, 0.0);
    // --- Draw the X edges
    for (int corner=0; corner<4; ++corner)
    {
        glPushMatrix();
        glTranslated(           minBBox[0]           ,
                (corner&1)?minBBox[1]:maxBBox[1],
                (corner&2)?minBBox[2]:maxBBox[2]);
        glRotatef(90,0,1,0);
        gluCylinder(_tube, r, r, maxBBox[0] - minBBox[0], 10, 10);
        glPopMatrix();
    }

    // --- Draw the Y edges
    for (int corner=0; corner<4; ++corner)
    {
        glPushMatrix();
        glTranslated((corner&1)?minBBox[0]:maxBBox[0],
                minBBox[1]           ,
                (corner&2)?minBBox[2]:maxBBox[2]);
        glRotatef(-90,1,0,0);
        gluCylinder(_tube, r, r, maxBBox[1] - minBBox[1], 10, 10);
        glPopMatrix();
    }

    // --- Draw the Z edges
    for (int corner=0; corner<4; ++corner)
    {
        glPushMatrix();
        glTranslated((corner&1)?minBBox[0]:maxBBox[0],
                (corner&2)?minBBox[1]:maxBBox[1],
                minBBox[2]           );
        gluCylinder(_tube, r, r, maxBBox[2] - minBBox[2], 10, 10);
        glPopMatrix();
    }
}


// ----------------------------------------------------------------------------------
// --- Draw a "plane" in wireframe. The "plane" is parallel to the XY axis
// --- of the main coordinate system
// ----------------------------------------------------------------------------------
void MultithreadGUI::DrawXYPlane(double zo, double xmin, double xmax, double ymin,
        double ymax, double step)
{
    register double x, y;

    Enable<GL_DEPTH_TEST> depth;

    glBegin(GL_LINES);
    for (x = xmin; x <= xmax; x += step)
    {
        glVertex3d(x, ymin, zo);
        glVertex3d(x, ymax, zo);
    }
    glEnd();

    glBegin(GL_LINES);
    for (y = ymin; y <= ymax; y += step)
    {
        glVertex3d(xmin, y, zo);
        glVertex3d(xmax, y, zo);
    }
    glEnd();
}


// ----------------------------------------------------------------------------------
// --- Draw a "plane" in wireframe. The "plane" is parallel to the XY axis
// --- of the main coordinate system
// ----------------------------------------------------------------------------------
void MultithreadGUI::DrawYZPlane(double xo, double ymin, double ymax, double zmin,
        double zmax, double step)
{
    register double y, z;
    Enable<GL_DEPTH_TEST> depth;

    glBegin(GL_LINES);
    for (y = ymin; y <= ymax; y += step)
    {
        glVertex3d(xo, y, zmin);
        glVertex3d(xo, y, zmax);
    }
    glEnd();

    glBegin(GL_LINES);
    for (z = zmin; z <= zmax; z += step)
    {
        glVertex3d(xo, ymin, z);
        glVertex3d(xo, ymax, z);
    }
    glEnd();

}


// ----------------------------------------------------------------------------------
// --- Draw a "plane" in wireframe. The "plane" is parallel to the XY axis
// --- of the main coordinate system
// ----------------------------------------------------------------------------------
void MultithreadGUI::DrawXZPlane(double yo, double xmin, double xmax, double zmin,
        double zmax, double step)
{
    register double x, z;
    Enable<GL_DEPTH_TEST> depth;

    glBegin(GL_LINES);
    for (x = xmin; x <= xmax; x += step)
    {
        glVertex3d(x, yo, zmin);
        glVertex3d(x, yo, zmax);
    }
    glEnd();

    glBegin(GL_LINES);
    for (z = zmin; z <= zmax; z += step)
    {
        glVertex3d(xmin, yo, z);
        glVertex3d(xmax, yo, z);
    }
    glEnd();
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void MultithreadGUI::DrawLogo()
{
    int w = 0;
    int h = 0;

    if (texLogo && texLogo->getImage())
    {
        h = texLogo->getImage()->getHeight();
        w = texLogo->getImage()->getWidth();
    }
    else return;

    Enable <GL_TEXTURE_2D> tex;
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-0.5, _W, -0.5, _H, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (texLogo)
        texLogo->bind();

    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2d(0.0, 0.0);
    glVertex3d((_W-w)/2, (_H-h)/2, 0.0);

    glTexCoord2d(1.0, 0.0);
    glVertex3d( _W-(_W-w)/2, (_H-h)/2, 0.0);

    glTexCoord2d(1.0, 1.0);
    glVertex3d( _W-(_W-w)/2, _H-(_H-h)/2, 0.0);

    glTexCoord2d(0.0, 1.0);
    glVertex3d((_W-w)/2, _H-(_H-h)/2, 0.0);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void MultithreadGUI::DisplayOBJs(bool shadowPass)
{

    Enable<GL_LIGHTING> light;
    Enable<GL_DEPTH_TEST> depth;

    glShadeModel(GL_SMOOTH);
    //glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColor4f(1,1,1,1);
    glDisable(GL_COLOR_MATERIAL);

    if (!initTexturesDone)
    {
//         std::cout << "-----------------------------------> initTexturesDone\n";
        //---------------------------------------------------
        simulation::getSimulation()->initTextures(groot);
        //---------------------------------------------------
        initTexturesDone = true;
    }

    {
        Node *visualRoot = simulation::getSimulation()->getVisualRoot();
        if (shadowPass)
        {
            getSimulation()->drawShadows(groot);
            getSimulation()->drawShadows(visualRoot);
        }
        else
        {
            getSimulation()->draw(&vparams,groot);
            getSimulation()->draw(&vparams,visualRoot);
        }
        if (_axis)
        {
            DrawAxis(0.0, 0.0, 0.0, 10.0);
            if (sceneMinBBox[0] < sceneMaxBBox[0])
            {
                Vec3d minTemp=sceneMinBBox;
                Vec3d maxTemp=sceneMaxBBox;
                DrawBox(minTemp.ptr(), maxTemp.ptr());
            }
        }
    }

    // glDisable(GL_COLOR_MATERIAL);
}

// -------------------------------------------------------
// ---
// -------------------------------------------------------
void MultithreadGUI::DisplayMenu(void)
{
    Disable<GL_LIGHTING> light;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-0.5, _W, -0.5, _H, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glColor3f(0.3f, 0.7f, 0.95f);
    glRasterPos2i(_W / 2 - 5, _H - 15);
    //sprintf(buffer,"FPS: %.1f\n", _frameRate.GetFPS());
    //PrintString(GLUT_BITMAP_HELVETICA_12, buffer);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void MultithreadGUI::DrawScene(void)
{

    _newQuat.buildRotationMatrix(_sceneTransform.rotation);
    calcProjection();

#ifdef SOFA_HAVE_GLEW
    if (_shadow)
    {
        //glGetDoublev(GL_MODELVIEW_MATRIX,lastModelviewMatrix);

        // Update the light matrices for it's current position
        StoreLightMatrices();

        // Set the current viewport to our texture size
        glViewport(0, 0, (int)SHADOW_WIDTH, (int)SHADOW_HEIGHT);

        // Clear the screen and depth buffer so we can render from the light's view
        glClearColor(0.0f,0.0f,0.0f,0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Now we just need to set the matrices for the light before we render
        glMatrixMode(GL_PROJECTION);

        // Push on a matrix to make sure we can restore to the old matrix easily
        glPushMatrix();
        {
            // Set the current projection matrix to our light's projection matrix
            glLoadMatrixf(g_mProjection);

            // Load modelview mode to set our light's modelview matrix
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            {
                // Load the light's modelview matrix before we render to a texture
                glLoadMatrixf(g_mModelView);

                // Since we don't care about color when rendering the depth values to
                // the shadow-map texture, we disable color writing to increase speed.
                glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);

                // This turns of the polygon offset functionality to fix artifacts.
                // Comment this out and run the program to see what artifacts I mean.
                glEnable(GL_POLYGON_OFFSET_FILL);
                glDisable(GL_BLEND);

                // Eliminate artifacts caused by shadow mapping
                //    glPolygonOffset(1.0f, 0.10f);
                glPolygonOffset(g_DepthOffset[0], g_DepthOffset[1]);

                _sceneTransform.Apply();
                // Render the world according to the light's view
                DisplayOBJs(true);

                // Now that the world is rendered, save the depth values to a texture
                glDisable(GL_BLEND);
                //glEnable(GL_TEXTURE_2D);
                glBindTexture(GL_TEXTURE_2D, g_DepthTexture);

                glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, (int)SHADOW_WIDTH, (int)SHADOW_HEIGHT);

                // We can turn color writing back on since we already stored the depth values
                glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

                // Turn off polygon offsetting
                glDisable(GL_POLYGON_OFFSET_FILL);

            } glPopMatrix();
            // Go back to the projection mode and restore the original matrix
            glMatrixMode(GL_PROJECTION);
            // Restore the original projection matrix
        } glPopMatrix();

        // Go back to modelview model to start drawing like normal
        glMatrixMode(GL_MODELVIEW);

        // Restore our normal viewport size to our screen width and height
        glViewport(0, 0, GetWidth(), GetHeight());

        // Clear the color and depth bits and start over from the camera's view
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        //glPushMatrix();{
        //    glLoadIdentity();
        //    _sceneTransform.ApplyInverse();
        //    glGetFloatv(GL_MODELVIEW_MATRIX, g_mCameraInverse);
        //}glPopMatrix();

        glLightfv( GL_LIGHT0, GL_POSITION, _lightPosition );
        /*
        {
            glEnable(GL_TEXTURE_2D);
            glActiveTextureARB(GL_TEXTURE0_ARB);
            glBindTexture(GL_TEXTURE_2D, g_Texture[SHADOW_ID]);
            glTexEnvi(GL_TEXTURE_2D,GL_TEXTURE_ENV_MODE,  GL_REPLACE);
            Disable<GL_DEPTH_TEST> dtoff;
            Disable<GL_LIGHTING> dlight;
            glColor3f(1,1,1);
            glViewport(0, 0, 128, 128);
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(0,1,0,1,-1,1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();{
                glLoadIdentity();
                glBegin(GL_QUADS);{
                    glTexCoord2f(0,0);
                    glVertex2f(0,0);
                    glTexCoord2f(0,1);
                    glVertex2f(0,1);
                    glTexCoord2f(1,1);
                    glVertex2f(1,1);
                    glTexCoord2f(1,0);
                    glVertex2f(1,0);
                }glEnd();
            }glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
            glViewport(0, 0, GetWidth(), GetHeight());
        }
        */


        // Render the world and apply the shadow map texture to it
        glMatrixMode(GL_PROJECTION);
        glLoadMatrixd(lastProjectionMatrix);
        glMatrixMode(GL_MODELVIEW);
        ApplyShadowMap();
        {
            // NICO
            Enable<GL_TEXTURE_2D> texture_on;
            glDisable(GL_BLEND);
            glBindTexture(GL_TEXTURE_2D, ShadowTextureMask);
            glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, GetWidth(), GetHeight());
        }
        if (_background==0)
            glClearColor(0.0589f, 0.0589f, 0.0589f, 1.0f);
        else if (_background==1)
            glClearColor(0.0f,0.0f,0.0f,0.0f);
        else if (_background==2)
            glClearColor(1.0f,1.0f,1.0f,1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (_background==0)
            DrawLogo();
        glPushMatrix();
        _sceneTransform.Apply();
        DisplayOBJs();
        glPopMatrix();

        {
            float ofu = GetWidth()/(float)SHADOW_MASK_SIZE;
            float ofv = GetHeight()/(float)SHADOW_MASK_SIZE;
            glActiveTextureARB(GL_TEXTURE0_ARB);
            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D,ShadowTextureMask);
            //glTexEnvi(GL_TEXTURE_2D,GL_TEXTURE_ENV_MODE,  GL_REPLACE);
            Disable<GL_DEPTH_TEST> dtoff;
            Disable<GL_LIGHTING> dlight;
            Enable<GL_BLEND> blend_on;
            glBlendFunc(GL_ZERO, GL_ONE_MINUS_SRC_COLOR);
            glColor3f(1,1,1);
            glViewport(0, 0, GetWidth(), GetHeight());
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(0,1,0,1,-1,1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            {
                glLoadIdentity();
                glBegin(GL_QUADS);
                {
                    glTexCoord2f(0,0);
                    glVertex2f(0,0);
                    glTexCoord2f(0,ofv);
                    glVertex2f(0,1);
                    glTexCoord2f(ofu,ofv);
                    glVertex2f(1,1);
                    glTexCoord2f(ofu,0);
                    glVertex2f(1,0);
                } glEnd();
            } glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
            glViewport(0, 0, GetWidth(), GetHeight());
            glDisable(GL_TEXTURE_2D);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        }

    }
    else
#endif
    {
        if (_background==0)
            DrawLogo();

        glLoadIdentity();
        _sceneTransform.Apply();

        glGetDoublev(GL_MODELVIEW_MATRIX,lastModelviewMatrix);

        if (_renderingMode == GL_RENDER)
        {
            // Initialize lighting
            glPushMatrix();
            glLoadIdentity();
            glLightfv(GL_LIGHT0, GL_POSITION, _lightPosition);
            glPopMatrix();
            Enable<GL_LIGHT0> light0;

            glColor3f(0.5f, 0.5f, 0.6f);
            //    DrawXZPlane(-4.0, -20.0, 20.0, -20.0, 20.0, 1.0);
            //    DrawAxis(0.0, 0.0, 0.0, 10.0);

            DisplayOBJs();

            DisplayMenu();        // always needs to be the last object being drawn
        }
    }
}


// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void MultithreadGUI::resizeGL(int width, int height)
{

    _W = width;
    _H = height;

//     std::cout << "GL window: " <<width<<"x"<<height <<std::endl;

    calcProjection();
}


// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void MultithreadGUI::calcProjection()
{
    int width = _W;
    int height = _H;
    double xNear, yNear, zNear, zFar, xOrtho, yOrtho;
    double xFactor = 1.0, yFactor = 1.0;
    double offset;
    double xForeground, yForeground, zForeground, xBackground, yBackground,
           zBackground;

    //if (!sceneBBoxIsValid)
    {
        getSimulation()->computeBBox(groot, sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        getSimulation()->computeBBox(getSimulation()->getVisualRoot(), sceneMinBBox.ptr(), sceneMaxBBox.ptr());
        sceneBBoxIsValid = true;
    }
    //std::cout << "Scene BBox = "<<sceneMinBBox<<" - "<<sceneMaxBBox<<"\n";
    if (sceneMinBBox[0] > sceneMaxBBox[0])
    {
        zNear = 1.0;
        zFar = 1000.0;
    }
    else
    {
        zNear = 1e10;
        zFar = -1e10;
        double minBBox[3] = {sceneMinBBox[0], sceneMinBBox[1], sceneMinBBox[2] };
        double maxBBox[3] = {sceneMaxBBox[0], sceneMaxBBox[1], sceneMaxBBox[2] };
        if (_axis)
        {
            for (int i=0; i<3; i++)
            {
                if (minBBox[i]>-2) minBBox[i] = -2;
                if (maxBBox[i]<14) maxBBox[i] = 14;
            }
        }

        for (int corner=0; corner<8; ++corner)
        {
            Vector3 p((corner&1)?minBBox[0]:maxBBox[0],
                    (corner&2)?minBBox[1]:maxBBox[1],
                    (corner&4)?minBBox[2]:maxBBox[2]);
            p = _sceneTransform * p;
            double z = -p[2];
            if (z < zNear) zNear = z;
            if (z > zFar) zFar = z;
        }
        if (zFar <= 0 || zFar >= 1000)
        {
            zNear = 1;
            zFar = 1000.0;
        }
        else
        {
            zNear *= 0.9; // add some margin
            zFar *= 1.1;
            if (zNear < zFar*0.01)
                zNear = zFar*0.01;
            if (zNear < 1.0) zNear = 1.0;
            if (zFar < 2.0) zFar = 2.0;
        }
        //std::cout << "Z = ["<<zNear<<" - "<<zFar<<"]\n";
    }
    xNear = 0.35*zNear;
    yNear = 0.35*zNear;
    offset = 0.001*zNear;        // for foreground and background planes

    xOrtho = fabs(_sceneTransform.translation[2]) * xNear / zNear;
    yOrtho = fabs(_sceneTransform.translation[2]) * yNear / zNear;

    if ((height != 0) && (width != 0))
    {
        if (height > width)
        {
            xFactor = 1.0;
            yFactor = (double) height / (double) width;
        }
        else
        {
            xFactor = (double) width / (double) height;
            yFactor = 1.0;
        }
    }

    lastViewport[0] = 0;
    lastViewport[1] = 0;
    lastViewport[2] = width;
    lastViewport[3] = height;
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    zForeground = -zNear - offset;
    zBackground = -zFar + offset;

    glFrustum(-xNear * xFactor, xNear * xFactor, -yNear * yFactor,
            yNear * yFactor, zNear, zFar);
    xForeground = -zForeground * xNear / zNear;
    yForeground = -zForeground * yNear / zNear;
    xBackground = -zBackground * xNear / zNear;
    yBackground = -zBackground * yNear / zNear;

    xForeground *= xFactor;
    yForeground *= yFactor;
    xBackground *= xFactor;
    yBackground *= yFactor;

    glGetDoublev(GL_PROJECTION_MATRIX,lastProjectionMatrix);

    glMatrixMode(GL_MODELVIEW);

    vparams.zFar()  = zFar;
    vparams.zNear() = zNear;
    vparams.viewport() = sofa::helper::make_array(0,0,width,height);
    vparams.sceneBBox() = sofa::defaulttype::BoundingBox(sceneMinBBox,sceneMaxBBox);
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void MultithreadGUI::paintGL()
{
    //    ctime_t beginDisplay;
    //ctime_t endOfDisplay;

    //    beginDisplay = MesureTemps();

    // valid() is turned off when FLTK creates a new context for this window
    // or when the window resizes, and is turned on after draw() is called.
    // Use this to avoid unneccessarily initializing the OpenGL context.
    //static double lastOrthoTransZ = 0.0;
    /*
    if (!valid())
    {
    InitGFX();        // this has to be called here since we don't know when the context is created
    _W = w();
    _H = h();
    reshape(_W, _H);
    }
    */
    // clear buffers (color and depth)
    if (_background==0)
        glClearColor(0.0589f, 0.0589f, 0.0589f, 1.0f);
    else if (_background==1)
        glClearColor(0.0f,0.0f,0.0f,0.0f);
    else if (_background==2)
        glClearColor(1.0f,1.0f,1.0f,1.0f);
    glClearDepth(1.0);
    glClear(_clearBuffer);

    // draw the scene
    DrawScene();

    if (_video)
    {
#ifdef CAPTURE_PERIOD
        static int counter = 0;
        if ((counter++ % CAPTURE_PERIOD)==0)
#endif
            screenshot(2);
    }
}

void MultithreadGUI::eventNewStep()
{
    static ctime_t beginTime[10];
    static const ctime_t timeTicks = CTime::getRefTicksPerSec();
    static int frameCounter = 0;
    if (frameCounter==0)
    {
        ctime_t t = CTime::getRefTime();
        for (int i=0; i<10; i++)
            beginTime[i] = t;
    }
    ++frameCounter;
    if ((frameCounter%10) == 0)
    {
        ctime_t curtime = CTime::getRefTime();
        int i = ((frameCounter/10)%10);
        double fps = ((double)timeTicks / (curtime - beginTime[i]))*(frameCounter<100?frameCounter:100);
        char buf[100];
        sprintf(buf, "%.1f FPS", fps);
        std::string title = "SOFA";
        if (!sceneFileName.empty())
        {
            title += " :: ";
            title += sceneFileName;
        }
        title += " :: ";
        title += buf;
        glutSetWindowTitle(title.c_str());

        beginTime[i] = curtime;
        //frameCounter = 0;
    }
    if (m_displayComputationTime && (frameCounter%100) == 0 && groot!=NULL)
    {
        std::cout << "========== ITERATION " << frameCounter << " ==========\n";
        const simulation::Node::NodeTimer& total = groot->getTotalTime();
        const std::map<std::string, simulation::Node::NodeTimer>& times = groot->getVisitorTime();
        const std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, simulation::Node::ObjectTimer> >& objtimes = groot->getObjectTime();
        const double fact = 1000000.0 / (100*groot->getTimeFreq());
        for (std::map<std::string, simulation::Node::NodeTimer>::const_iterator it = times.begin(); it != times.end(); ++it)
        {
            std::cout << "TIME "<<it->first<<": " << ((int)(fact*it->second.tTree+0.5))*0.001 << " ms (" << (1000*it->second.tTree/total.tTree)*0.1 << " %).\n";
            std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, simulation::Node::ObjectTimer> >::const_iterator it1 = objtimes.find(it->first);
            if (it1 != objtimes.end())
            {
                for (std::map<sofa::core::objectmodel::BaseObject*, simulation::Node::ObjectTimer>::const_iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2)
                {
                    std::cout << "  "<< sofa::helper::gettypename(typeid(*(it2->first)))<<" "<< it2->first->getName() <<": "
                            << ((int)(fact*it2->second.tObject+0.5))*0.001 << " ms (" << (1000*it2->second.tObject/it->second.tTree)*0.1 << " %).\n";
                }
            }
        }
        for (std::map<std::string, std::map<sofa::core::objectmodel::BaseObject*, simulation::Node::ObjectTimer> >::const_iterator it = objtimes.begin(); it != objtimes.end(); ++it)
        {
            if (times.count(it->first)>0) continue;
            ctime_t ttotal = 0;
            for (std::map<sofa::core::objectmodel::BaseObject*, simulation::Node::ObjectTimer>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                ttotal += it2->second.tObject;
            std::cout << "TIME "<<it->first<<": " << ((int)(fact*ttotal+0.5))*0.001 << " ms (" << (1000*ttotal/total.tTree)*0.1 << " %).\n";
            if (ttotal > 0)
                for (std::map<sofa::core::objectmodel::BaseObject*, simulation::Node::ObjectTimer>::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2)
                {
                    std::cout << "  "<< sofa::helper::gettypename(typeid(*(it2->first)))<<" "<< it2->first->getName() <<": "
                            << ((int)(fact*it2->second.tObject+0.5))*0.001 << " ms (" << (1000*it2->second.tObject/ttotal)*0.1 << " %).\n";
                }
        }
        std::cout << "TOTAL TIME: " << ((int)(fact*total.tTree+0.5))*0.001 << " ms (" << ((int)(100/(fact*total.tTree*0.000001)+0.5))*0.01 << " FPS).\n";
        groot->resetTime();
    }
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void MultithreadGUI::animate(void)
{
    if (_spinning)
    {
        _newQuat = _currentQuat + _newQuat;
    }

    // update the entire scene
    redraw();
}


// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void MultithreadGUI::ApplySceneTransformation(int x, int y)
{
    float    x1, x2, y1, y2;
    float    xshift, yshift, zshift;

    if (_moving)
    {
        if (_navigationMode == TRACKBALL_MODE)
        {
            x1 = (2.0f * _W / 2.0f - _W) / _W;
            y1 = (_H - 2.0f * _H / 2.0f) / _H;
            x2 = (2.0f * (x + (-_mouseX + _W / 2.0f)) - _W) / _W;
            y2 = (_H - 2.0f * (y + (-_mouseY + _H / 2.0f))) / _H;
            _currentTrackball.ComputeQuaternion(x1, y1, x2, y2);
            _currentQuat = _currentTrackball.GetQuaternion();
            _savedMouseX = _mouseX;
            _savedMouseY = _mouseY;
            _mouseX = x;
            _mouseY = y;
            _newQuat = _currentQuat + _newQuat;
            redraw();
        }
        else if (_navigationMode == ZOOM_MODE)
        {
            zshift = (2.0f * y - _W) / _W - (2.0f * _mouseY - _W) / _W;
            _sceneTransform.translation[2] = _previousEyePos[2] -
                    _zoomSpeed * zshift;
            redraw();
        }
        else if (_navigationMode == PAN_MODE)
        {
            xshift = (2.0f * x - _W) / _W - (2.0f * _mouseX - _W) / _W;
            yshift = (2.0f * y - _W) / _W - (2.0f * _mouseY - _W) / _W;
            _sceneTransform.translation[0] = _previousEyePos[0] +
                    _panSpeed * xshift;
            _sceneTransform.translation[1] = _previousEyePos[1] -
                    _panSpeed * yshift;
            redraw();
        }
    }
}


// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void MultithreadGUI::ApplyMouseInteractorTransformation(int x, int y)
{
    // Mouse Interaction
    double coeffDeplacement = 0.025;
    Quaternion conjQuat, resQuat, _newQuatBckUp;

    float x1, x2, y1, y2;

    if (_mouseInteractorMoving)
    {
        if (_mouseInteractorRotationMode)
        {
            if ((_mouseInteractorSavedPosX != x) || (_mouseInteractorSavedPosY != y))
            {
                x1 = 0;
                y1 = 0;
                x2 = (2.0f * (x + (-_mouseInteractorSavedPosX + _W / 2.0f)) - _W) / _W;
                y2 = (_H - 2.0f * (y + (-_mouseInteractorSavedPosY + _H / 2.0f))) / _H;

                _mouseInteractorTrackball.ComputeQuaternion(x1, y1, x2, y2);
                _mouseInteractorCurrentQuat = _mouseInteractorTrackball.GetQuaternion();
                _mouseInteractorSavedPosX = x;
                _mouseInteractorSavedPosY = y;

                _mouseInteractorNewQuat = _mouseInteractorCurrentQuat + _mouseInteractorNewQuat;
                _mouseRotate = true;
            }
            else
            {
                _mouseRotate = false;
            }

            redraw();
        }
        else if (_mouseInteractorTranslationMode)
        {
            _mouseInteractorAbsolutePosition =  Vector3(0,0,0);
            _mouseInteractorRelativePosition =  Vector3(0,0,0);

            if (_translationMode == XY_TRANSLATION)
            {
                _mouseInteractorAbsolutePosition[0] = coeffDeplacement * (x - _mouseInteractorSavedPosX);
                _mouseInteractorAbsolutePosition[1] = -coeffDeplacement * (y - _mouseInteractorSavedPosY);

                _mouseInteractorSavedPosX = x;
                _mouseInteractorSavedPosY = y;
            }
            else if (_translationMode == Z_TRANSLATION)
            {
                _mouseInteractorAbsolutePosition[2] = coeffDeplacement * (y - _mouseInteractorSavedPosY);

                _mouseInteractorSavedPosX = x;
                _mouseInteractorSavedPosY = y;
            }

            _newQuatBckUp[0] = _newQuat[0];
            _newQuatBckUp[1] = _newQuat[1];
            _newQuatBckUp[2] = _newQuat[2];
            _newQuatBckUp[3] = _newQuat[3];

            _newQuatBckUp.normalize();

            // Conjugate calculation of the scene orientation quaternion
            conjQuat[0] = -_newQuatBckUp[0];
            conjQuat[1] = -_newQuatBckUp[1];
            conjQuat[2] = -_newQuatBckUp[2];
            conjQuat[3] = _newQuatBckUp[3];

            conjQuat.normalize();

            resQuat = _newQuatBckUp.quatVectMult(_mouseInteractorAbsolutePosition) * conjQuat;

            _mouseInteractorRelativePosition[0] = resQuat[0];
            _mouseInteractorRelativePosition[1] = resQuat[1];
            _mouseInteractorRelativePosition[2] = resQuat[2];

            _mouseTrans = true;
            redraw();
        }
    }
}


// ----------------------------------------
// --- Handle events (mouse, keyboard, ...)
// ----------------------------------------

bool MultithreadGUI::isControlPressed() const
{
    return m_isControlPressed;
    //return glutGetModifiers()&GLUT_ACTIVE_CTRL;
}

bool MultithreadGUI::isShiftPressed() const
{
    return m_isShiftPressed;
    //return glutGetModifiers()&GLUT_ACTIVE_SHIFT;
}

bool MultithreadGUI::isAltPressed() const
{
    return m_isAltPressed;
    //return glutGetModifiers()&GLUT_ACTIVE_ALT;
}

void MultithreadGUI::updateModifiers()
{
    m_isControlPressed =  (glutGetModifiers()&GLUT_ACTIVE_CTRL )!=0;
    m_isShiftPressed   =  (glutGetModifiers()&GLUT_ACTIVE_SHIFT)!=0;
    m_isAltPressed     =  (glutGetModifiers()&GLUT_ACTIVE_ALT  )!=0;
}

void MultithreadGUI::keyPressEvent ( int k )
{
    if( isControlPressed() ) // pass event to the scene data structure
    {
        //cerr<<"MultithreadGUI::keyPressEvent, key = "<<k<<" with Control pressed "<<endl;
        sofa::core::objectmodel::KeypressedEvent keyEvent(k);
        groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
    }
    else  // control the GUI
        switch(k)
        {

        case 's':
            // --- save screenshot
        {
            screenshot();
            break;
        }
        case 'v':
            // --- save video
        {
            _video = !_video;
            capture.setCounter();
            break;
        }
        case 'w':
            // --- save current view
        {
            saveView();
            break;
        }

        case 'o':
            // --- export to OBJ
        {
            exportOBJ();
            break;
        }
        case 'p':
            // --- export to a succession of OBJ to make a video
        {
            _animationOBJ = !_animationOBJ;
            _animationOBJcounter = 0;
            break;
        }
        case  'r':
            // --- draw axis
        {
            _axis = !_axis;
            redraw();
            break;
        }
        case 'b':
            // --- change background
        {
            _background = (_background+1)%3;
            redraw();
            break;
        }

        case 'l':
            // --- draw shadows
        {
            if (_glshadow)
            {
                _shadow = !_shadow;
                redraw();
            }
            else
            {
                printf("WARNING MultithreadGUI : shadows are not supported !\n");
                _shadow=false;
            }
            break;
        }

        case ' ':
            // --- start/stop
        {
            playpause();
            break;
        }

        case 'n':
            // --- step
        {
            step();
            redraw();
            break;
        }

        case 'q': //GLUT_KEY_Escape:
        {
            exit(0);
            break;
        }

        case 'c':
        {
            // --- switch interaction mode
            if (!_mouseInteractorTranslationMode)
            {
                std::cout << "Interaction Mode ON\n";
                _mouseInteractorTranslationMode = true;
                _mouseInteractorRotationMode = false;
            }
            else
            {
                std::cout << "Interaction Mode OFF\n";
                _mouseInteractorTranslationMode = false;
                _mouseInteractorRotationMode = false;
            }
            break;
        }
        case GLUT_KEY_F5:
        {
            if (!sceneFileName.empty())
            {
                std::cout << "Reloading "<<sceneFileName<<std::endl;
                std::string filename = sceneFileName;
                Quaternion q = _newQuat;
                Transformation t = _sceneTransform;
                simulation::Node* newroot = getSimulation()->load(filename.c_str());
                getSimulation()->init(newroot);
                if (newroot == NULL)
                {
                    std::cerr << "Failed to load "<<filename<<std::endl;
                    break;
                }
                setScene(newroot, filename.c_str());
                _newQuat = q;
                _sceneTransform = t;
            }

            break;
        }
        }
}


void MultithreadGUI::keyReleaseEvent ( int k )
{
    //cerr<<"MultithreadGUI::keyReleaseEvent, key = "<<k<<endl;
    if( isControlPressed() ) // pass event to the scene data structure
    {
        sofa::core::objectmodel::KeyreleasedEvent keyEvent(k);
        groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
    }
}

// ---------------------- Here are the mouse controls for the scene  ----------------------
void MultithreadGUI::mouseEvent ( int type, int eventX, int eventY, int button )
{

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT,viewport);

    MousePosition mousepos;
    mousepos.screenWidth  = viewport[2];
    mousepos.screenHeight = viewport[3];
    mousepos.x      = eventX;
    mousepos.y      = eventY;

    if( isShiftPressed() )
    {
        pick.activateRay(viewport[2],viewport[3]);
    }
    else
    {
        pick.deactivateRay();
    }
    if (_mouseInteractorRotationMode)
    {
        switch (type)
        {
        case MouseButtonPress:
            // Mouse left button is pushed
            if (button == GLUT_LEFT_BUTTON)
            {
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            break;

        case MouseMove:
            //
            break;

        case MouseButtonRelease:
            // Mouse left button is released
            if (button == GLUT_LEFT_BUTTON)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            break;

        default:
            break;
        }
        ApplyMouseInteractorTransformation(eventX, eventY);
    }
    else if (_mouseInteractorTranslationMode)
    {
        switch (type)
        {
        case MouseButtonPress:
            // Mouse left button is pushed
            if (button == GLUT_LEFT_BUTTON)
            {
                _translationMode = XY_TRANSLATION;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
                _mouseInteractorMoving = true;
            }
            // Mouse right button is pushed
            else if (button == GLUT_RIGHT_BUTTON)
            {
                _translationMode = Z_TRANSLATION;
                _mouseInteractorSavedPosY = eventY;
                _mouseInteractorMoving = true;
            }

            break;

        case MouseButtonRelease:
            // Mouse left button is released
            if ((button == GLUT_LEFT_BUTTON) && (_translationMode == XY_TRANSLATION))
            {
                if (_mouseInteractorMoving)
                {
                    //_mouseInteractorRelativePosition = Vector3::ZERO;
                    _mouseInteractorMoving = false;
                }
            }
            // Mouse right button is released
            else if ((button == GLUT_RIGHT_BUTTON) && (_translationMode == Z_TRANSLATION))
            {
                if (_mouseInteractorMoving)
                {
                    //_mouseInteractorRelativePosition = Vector3::ZERO;
                    _mouseInteractorMoving = false;
                }
            }
            break;

        default:
            break;
        }

        ApplyMouseInteractorTransformation(eventX, eventY);
    }
    else if (isShiftPressed())
    {
        _moving = false;

        Vec3d p0, px, py, pz;
        gluUnProject(eventX, lastViewport[3]-1-(eventY), 0, lastModelviewMatrix, lastProjectionMatrix, lastViewport, &(p0[0]), &(p0[1]), &(p0[2]));
        gluUnProject(eventX+1, lastViewport[3]-1-(eventY), 0, lastModelviewMatrix, lastProjectionMatrix, lastViewport, &(px[0]), &(px[1]), &(px[2]));
        gluUnProject(eventX, lastViewport[3]-1-(eventY+1), 0, lastModelviewMatrix, lastProjectionMatrix, lastViewport, &(py[0]), &(py[1]), &(py[2]));
        gluUnProject(eventX, lastViewport[3]-1-(eventY), 1, lastModelviewMatrix, lastProjectionMatrix, lastViewport, &(pz[0]), &(pz[1]), &(pz[2]));
        px -= p0;
        py -= p0;
        pz -= p0;
        px.normalize();
        py.normalize();
        pz.normalize();
        Mat4x4d transform;
        transform.identity();
        transform[0][0] = px[0];
        transform[1][0] = px[1];
        transform[2][0] = px[2];
        transform[0][1] = py[0];
        transform[1][1] = py[1];
        transform[2][1] = py[2];
        transform[0][2] = pz[0];
        transform[1][2] = pz[1];
        transform[2][2] = pz[2];
        transform[0][3] = p0[0];
        transform[1][3] = p0[1];
        transform[2][3] = p0[2];
        Mat3x3d mat; mat = transform;
        Quat q; q.fromMatrix(mat);

        Vec3d position, direction;
        position  = transform*Vec4d(0,0,0,1);
        direction = transform*Vec4d(0,0,1,0);
        direction.normalize();
        pick.updateRay(position, direction);
        pick.updateMouse2D(mousepos);
        switch (type)
        {
        case MouseButtonPress:
            if (button == GLUT_LEFT_BUTTON) // Shift+Leftclick to deform the mesh
            {
                pick.handleMouseEvent(PRESSED, LEFT);
            }
            else if (button == GLUT_RIGHT_BUTTON) // Shift+Rightclick to remove triangles
            {
                pick.handleMouseEvent(PRESSED, RIGHT);
            }
            else if (button == GLUT_MIDDLE_BUTTON) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
            {
                pick.handleMouseEvent(PRESSED, MIDDLE);
            }
            break;
        case MouseButtonRelease:
            //if (button == GLUT_LEFT_BUTTON)
        {
            if (button == GLUT_LEFT_BUTTON) // Shift+Leftclick to deform the mesh
            {
                pick.handleMouseEvent(RELEASED, LEFT);
            }
            else if (button == GLUT_RIGHT_BUTTON) // Shift+Rightclick to remove triangles
            {
                pick.handleMouseEvent(RELEASED, RIGHT);
            }
            else if (button == GLUT_MIDDLE_BUTTON) // Shift+Midclick (by 2 steps defining 2 input points) to cut from one point to another
            {
                pick.handleMouseEvent(RELEASED, MIDDLE);
            }
        }
        break;
        default: break;
        }
    }
    else if (isAltPressed())
    {
        _moving = false;
        switch (type)
        {
        case MouseButtonPress:
            // Mouse left button is pushed
            if (button == GLUT_LEFT_BUTTON)
            {
                _navigationMode = BTLEFT_MODE;
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            // Mouse right button is pushed
            else if (button == GLUT_RIGHT_BUTTON)
            {
                _navigationMode = BTRIGHT_MODE;
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            // Mouse middle button is pushed
            else if (button == GLUT_MIDDLE_BUTTON)
            {
                _navigationMode = BTMIDDLE_MODE;
                _mouseInteractorMoving = true;
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
            break;

        case MouseMove:
            //
            break;

        case MouseButtonRelease:
            // Mouse left button is released
            if (button == GLUT_LEFT_BUTTON)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            // Mouse right button is released
            else if (button == GLUT_RIGHT_BUTTON)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            // Mouse middle button is released
            else if (button == GLUT_MIDDLE_BUTTON)
            {
                if (_mouseInteractorMoving)
                {
                    _mouseInteractorMoving = false;
                }
            }
            break;

        default:
            break;
        }
        if (_mouseInteractorMoving && _navigationMode == BTLEFT_MODE)
        {
            int dx = eventX - _mouseInteractorSavedPosX;
            int dy = eventY - _mouseInteractorSavedPosY;
            if (dx || dy)
            {
                _lightPosition[0] -= dx*0.1;
                _lightPosition[1] += dy*0.1;
                std::cout << "Light = "<< _lightPosition[0] << " "<< _lightPosition[1] << " "<< _lightPosition[2] << std::endl;
                redraw();
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
        }
        else if (_mouseInteractorMoving && _navigationMode == BTMIDDLE_MODE)
        {
            int dx = eventX - _mouseInteractorSavedPosX;
            int dy = eventY - _mouseInteractorSavedPosY;
            if (dx || dy)
            {
                //g_DepthBias[0] += dx*0.01;
                g_DepthBias[1] += dy*0.01;
                std::cout << "Depth bias = "<< g_DepthBias[0] << " " << g_DepthBias[1] << std::endl;
                redraw();
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
        }
        else if (_mouseInteractorMoving && _navigationMode == BTRIGHT_MODE)
        {
            int dx = eventX - _mouseInteractorSavedPosX;
            int dy = eventY - _mouseInteractorSavedPosY;
            if (dx || dy)
            {
                g_DepthOffset[0] += dx*0.01;
                g_DepthOffset[1] += dy*0.01;
                std::cout << "Depth offset = "<< g_DepthOffset[0] << " " << g_DepthOffset[1] << std::endl;
                redraw();
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
        }
    }
    else if (isControlPressed())
    {
        std::vector< sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* > instruments;
        groot->getTreeObjects<sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>, std::vector< sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* > >(&instruments);
        //std::cout << instruments.size() << " instruments\n";
        if (!instruments.empty())
        {
            _moving = false;
            sofa::core::behavior::MechanicalState<sofa::defaulttype::LaparoscopicRigidTypes>* instrument = instruments[0];
            switch (type)
            {
            case MouseButtonPress:
                // Mouse left button is pushed
                if (button == GLUT_LEFT_BUTTON)
                {
                    _navigationMode = BTLEFT_MODE;
                    _mouseInteractorMoving = true;
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
                // Mouse right button is pushed
                else if (button == GLUT_RIGHT_BUTTON)
                {
                    _navigationMode = BTRIGHT_MODE;
                    _mouseInteractorMoving = true;
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
                // Mouse middle button is pushed
                else if (button == GLUT_MIDDLE_BUTTON)
                {
                    _navigationMode = BTMIDDLE_MODE;
                    _mouseInteractorMoving = true;
                    _mouseInteractorSavedPosX = eventX;
                    _mouseInteractorSavedPosY = eventY;
                }
                break;

            case MouseMove:
                //
                break;

            case MouseButtonRelease:
                // Mouse left button is released
                if (button == GLUT_LEFT_BUTTON)
                {
                    if (_mouseInteractorMoving)
                    {
                        _mouseInteractorMoving = false;
                    }
                }
                // Mouse right button is released
                else if (button == GLUT_RIGHT_BUTTON)
                {
                    if (_mouseInteractorMoving)
                    {
                        _mouseInteractorMoving = false;
                    }
                }
                // Mouse middle button is released
                else if (button == GLUT_MIDDLE_BUTTON)
                {
                    if (_mouseInteractorMoving)
                    {
                        _mouseInteractorMoving = false;
                    }
                }
                break;

            default:
                break;
            }
            if (_mouseInteractorMoving)
            {
                {
                    helper::WriteAccessor<Data<sofa::defaulttype::LaparoscopicRigidTypes::VecCoord> > instrumentX = *instrument->write(core::VecCoordId::position());
                    if (_navigationMode == BTLEFT_MODE)
                    {
                        int dx = eventX - _mouseInteractorSavedPosX;
                        int dy = eventY - _mouseInteractorSavedPosY;
                        if (dx || dy)
                        {
                            instrumentX[0].getOrientation() = instrumentX[0].getOrientation() * Quat(Vector3(0,1,0),dx*0.001) * Quat(Vector3(0,0,1),dy*0.001);
                            redraw();
                            _mouseInteractorSavedPosX = eventX;
                            _mouseInteractorSavedPosY = eventY;
                        }
                    }
                    else if (_navigationMode == BTMIDDLE_MODE)
                    {
                        int dx = eventX - _mouseInteractorSavedPosX;
                        int dy = eventY - _mouseInteractorSavedPosY;
                        if (dx || dy)
                        {
                            if (!groot || !groot->getContext()->getAnimate())
                                redraw();
                            _mouseInteractorSavedPosX = eventX;
                            _mouseInteractorSavedPosY = eventY;
                        }
                    }
                    else if (_navigationMode == BTRIGHT_MODE)
                    {
                        int dx = eventX - _mouseInteractorSavedPosX;
                        int dy = eventY - _mouseInteractorSavedPosY;
                        if (dx || dy)
                        {
                            instrumentX[0].getTranslation() += (dy)*0.01;
                            instrumentX[0].getOrientation() = instrumentX[0].getOrientation() * Quat(Vector3(1,0,0),dx*0.001);
                            if (!groot || !groot->getContext()->getAnimate())
                                redraw();
                            _mouseInteractorSavedPosX = eventX;
                            _mouseInteractorSavedPosY = eventY;
                        }
                    }
                }
                sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor(core::MechanicalParams::defaultInstance()).execute(instrument->getContext());
                sofa::simulation::UpdateMappingVisitor(core::ExecParams::defaultInstance()).execute(instrument->getContext());
                //static_cast<sofa::simulation::Node*>(instrument->getContext())->execute<sofa::simulation::MechanicalPropagatePositionAndVelocityVisitor>();
                //static_cast<sofa::simulation::Node*>(instrument->getContext())->execute<sofa::simulation::UpdateMappingVisitor>();
            }
        }
    }
    else
    {
        switch (type)
        {
        case MouseButtonPress:
            // rotate with left button
            if (button == GLUT_LEFT_BUTTON)
            {
                _navigationMode = TRACKBALL_MODE;
                _newTrackball.ComputeQuaternion(0.0, 0.0, 0.0, 0.0);
                _currentQuat = _newTrackball.GetQuaternion();
                _moving = true;
                _spinning = false;
                _mouseX = eventX;
                _mouseY = eventY;
            }
            // translate with middle button (if it exists)
            else if (button == GLUT_MIDDLE_BUTTON)
            {
                _navigationMode = PAN_MODE;
                _moving = true;
                _spinning = false;
                _mouseX = eventX;
                _mouseY = eventY;
                _previousEyePos[0] = _sceneTransform.translation[0];
                _previousEyePos[1] = _sceneTransform.translation[1];
            }
            // zoom with right button
            else if (button == GLUT_RIGHT_BUTTON)
            {
                _navigationMode = ZOOM_MODE;
                _moving = true;
                _spinning = false;
                _mouseX = eventX;
                _mouseY = eventY;
                _previousEyePos[2] = _sceneTransform.translation[2];
            }
            break;

        case MouseMove:
            //
            break;

        case MouseButtonRelease:
            // Mouse left button is released
            if (button == GLUT_LEFT_BUTTON)
            {
                if (_moving && _navigationMode == TRACKBALL_MODE)
                {
                    _moving = false;
                    int dx = eventX - _savedMouseX;
                    int dy = eventY - _savedMouseY;
                    if ((dx >= MINMOVE) || (dx <= -MINMOVE) ||
                        (dy >= MINMOVE) || (dy <= -MINMOVE))
                    {
                        _spinning = true;
                    }
                }
            }
            // Mouse middle button is released
            else if (button == GLUT_MIDDLE_BUTTON)
            {
                if (_moving && _navigationMode == PAN_MODE)
                {
                    _moving = false;
                }
            }
            // Mouse right button is released
            else if (button == GLUT_RIGHT_BUTTON)
            {
                if (_moving && _navigationMode == ZOOM_MODE)
                {
                    _moving = false;
                }
            }

            break;
            /*
            case FL_MOUSEWHEEL:
            // it is also possible to zoom with mouse wheel (if it exists)
            if (Fl::event_button() == FL_MOUSEWHEEL)
            {
            _navigationMode = ZOOM_MODE;
            _moving = true;
            _mouseX = 0;
            _mouseY = 0;
            eventX = 0;
            eventY = 10 * Fl::event_dy();
            _previousEyePos[2] = _sceneTransform.translation[2];
            }
            break;
            */
        default:
            break;
        }

        ApplySceneTransformation(eventX, eventY);
    }
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void MultithreadGUI::SwitchToPresetView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName+".view";
        std::ifstream in(viewFileName.c_str());
        if (!in.fail())
        {
            in >> _sceneTransform.translation[0];
            in >> _sceneTransform.translation[1];
            in >> _sceneTransform.translation[2];
            in >> _newQuat[0];
            in >> _newQuat[1];
            in >> _newQuat[2];
            in >> _newQuat[3];
            _newQuat.normalize();
            in.close();
            return;
        }
//        std::cout << "PRESET FAILED " << viewFileName << std::endl;
    }
//    std::cout << "PRESET" << std::endl;
    _sceneTransform.translation[0] = 0.0;
    _sceneTransform.translation[1] = 0.0;
    _sceneTransform.translation[2] = -50.0;
    _newQuat[0] = 0.17;
    _newQuat[1] = -0.83;
    _newQuat[2] = -0.26;
    _newQuat[3] = -0.44;
    //ResetScene();
}


void MultithreadGUI::step()
{
    {
        //groot->setLogTime(true);
#ifdef SOFA_SMP
        mg->step();
#else
        getSimulation()->animate(groot);
#endif
        getSimulation()->updateVisual(getSimulation()->getVisualRoot());

        if( m_dumpState )
            getSimulation()->dumpState( groot, *m_dumpStateStream );
        if( m_exportGnuplot )
            getSimulation()->exportGnuplot( groot, groot->getTime() );

        eventNewStep();
    }

    if (_animationOBJ)
    {
#ifdef CAPTURE_PERIOD
        static int counter = 0;
        if ((counter++ % CAPTURE_PERIOD)==0)
#endif
        {
            exportOBJ(false);
            ++_animationOBJcounter;
        }
    }
}

void MultithreadGUI::playpause()
{
    if (groot)
    {
        groot->getContext()->setAnimate(!groot->getContext()->getAnimate());
    }
}

void MultithreadGUI::dumpState(bool value)
{
    m_dumpState = value;
    if( m_dumpState )
    {
        m_dumpStateStream = new std::ofstream("dumpState.data");
    }
    else if( m_dumpStateStream!=NULL )
    {
        delete m_dumpStateStream;
        m_dumpStateStream = 0;
    }
}

void MultithreadGUI::displayComputationTime(bool value)
{
    m_displayComputationTime = value;
    if (groot)
    {
        groot->setLogTime(m_displayComputationTime);
    }
}

void MultithreadGUI::resetScene()
{
    if (groot)
    {
        getSimulation()->reset(groot);
        getSimulation()->reset(getSimulation()->getVisualRoot());
        redraw();
    }
}

void MultithreadGUI::resetView()
{
    SwitchToPresetView();
    redraw();
}

void MultithreadGUI::saveView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName+".view";
        std::ofstream out(viewFileName.c_str());
        if (!out.fail())
        {
            out << _sceneTransform.translation[0] << " " << _sceneTransform.translation[1] << " " << _sceneTransform.translation[2] << "\n";
            out << _newQuat[0] << " " << _newQuat[1] << " " << _newQuat[2] << " " << _newQuat[3] << "\n";
            out.close();
        }
        std::cout << "View parameters saved in "<<viewFileName<<std::endl;
    }
}

void MultithreadGUI::showVisual(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowVisualModels(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::showBehavior(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowBehaviorModels(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::showCollision(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowCollisionModels(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::showBoundingCollision(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowBoundingCollisionModels(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::showMapping(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowMappings(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::showMechanicalMapping(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowMechanicalMappings(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::showForceField(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowForceFields(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::showInteractionForceField(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowInteractionForceFields(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::showWireFrame(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowWireFrame(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::showNormals(bool value)
{
    if (groot)
    {
        groot->getContext()->setShowNormals(value);
        getSimulation()->updateVisualContext(groot);
    }
    redraw();
}

void MultithreadGUI::screenshot(int compression_level)
{
    capture.saveScreen(compression_level);
}

void MultithreadGUI::exportOBJ(bool exportMTL)
{
    if (!groot) return;
    std::ostringstream ofilename;
    if (!sceneFileName.empty())
    {
        const char* begin = sceneFileName.c_str();
        const char* end = strrchr(begin,'.');
        if (!end) end = begin + sceneFileName.length();
        ofilename << std::string(begin, end);
    }
    else
        ofilename << "scene";
//     double time = groot->getTime();
//     ofilename << '-' << (int)(time*1000);

    std::stringstream oss;
    oss.width(5);
    oss.fill('0');
    oss << _animationOBJcounter;

    ofilename << '_' << (oss.str().c_str());
    ofilename << ".obj";
    std::string filename = ofilename.str();
    std::cout << "Exporting OBJ Scene "<<filename<<std::endl;
    getSimulation()->exportOBJ(simulation::getSimulation()->getVisualRoot(), filename.c_str(),exportMTL);
}

void MultithreadGUI::setScene(sofa::simulation::Node* scene, const char* filename, bool)
{
    std::ostringstream ofilename;

    sceneFileName = (filename==NULL)?"":filename;
    if (!sceneFileName.empty())
    {
        const char* begin = sceneFileName.c_str();
        const char* end = strrchr(begin,'.');
        if (!end) end = begin + sceneFileName.length();
        ofilename << std::string(begin, end);
        ofilename << "_";
    }
    else
        ofilename << "scene_";

    capture.setPrefix(ofilename.str());
    groot = scene;
    initTexturesDone = false;
    sceneBBoxIsValid = false;
    redraw();
    pick.reset();
}

void MultithreadGUI::setExportGnuplot( bool exp )
{
    m_exportGnuplot = exp;
    if( m_exportGnuplot )
    {
        getSimulation()->initGnuplot( groot );
        getSimulation()->exportGnuplot( groot, groot->getTime() );
    }
}

} // namespace glut

} // namespace gui

} // namespace sofa
