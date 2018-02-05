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
#include "SimpleGUI.h"
#include <sofa/helper/system/config.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/simulation/Simulation.h>
#include <sofa/simulation/MechanicalVisitor.h>
#include <sofa/simulation/UpdateMappingVisitor.h>
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

#include <sofa/helper/system/thread/CTime.h>

#include <sofa/defaulttype/RigidTypes.h>
#include <sofa/defaulttype/BoundingBox.h>
#include <sofa/core/ObjectFactory.h>

#include <sofa/gui/OperationFactory.h>
#include <sofa/gui/MouseOperations.h>

#include <sofa/simulation/PropagateEventVisitor.h>
#ifdef SOFA_SMP
#include <SofaBaseVisual/VisualModelImpl.h>
#include <sofa/simulation/AnimateBeginEvent.h>
#include <sofa/simulation/CollisionVisitor.h>
#include <sofa/simulation/AnimateEndEvent.h>
#include <sofa/simulation/PropagateEventVisitor.h>
#include <sofa/simulation/VisualVisitor.h>
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
        // TODO SimpleGUI::instance->getScene()->execute<CollisionVisitor>();
        // TODO AnimateBeginEvent ev ( 0.0 );
        // TODO PropagateEventVisitor act ( &ev );
        // TODO SimpleGUI::instance->getScene()->execute ( act );
        //	sofa::simulation::tree::getSimulation()->animate(groot.get());

    }
};
struct animateTask
{
    void operator()()
    {
        //	std::cout << "Recording simulation with base name: " << writeSceneName << "\n";

        getSimulation()->animate( SimpleGUI::instance->getScene());

    }
};

struct collideTask
{
    void operator()()
    {
        //	std::cout << "Recording simulation with base name: " << writeSceneName << "\n";

        //   a1::Fork<doCollideTask>()();
        //	sofa::simulation::tree::getSimulation()->animate(groot.get());

    }
};
struct visuTask
{
    void operator()()
    {
        //	std::cout << "Recording simulation with base name: " << writeSceneName << "\n";
        // TODO AnimateEndEvent ev ( 0.0 );
        // TODO PropagateEventVisitor act ( &ev );
        // TODO SimpleGUI::instance->getScene()->execute ( act );
        // TODO SimpleGUI::instance->getScene()->execute<VisualUpdateVisitor>();

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

SimpleGUI* SimpleGUI::instance = NULL;

int SimpleGUI::mainLoop()
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
    glutMainLoop();
    return 0;
}

void SimpleGUI::redraw()
{
    glutPostRedisplay();
}

int SimpleGUI::closeGUI()
{
    delete this;
    return 0;
}


SOFA_DECL_CLASS(SimpleGUI)

static sofa::core::ObjectFactory::ClassEntry::SPtr classVisualModel;

int SimpleGUI::InitGUI(const char* /*name*/, const std::vector<std::string>& /*options*/)
{
    // Replace generic visual models with OglModel
    sofa::core::ObjectFactory::AddAlias("VisualModel", "OglModel", true,
            &classVisualModel);
    return 0;
}

BaseGUI* SimpleGUI::CreateGUI(const char* /*name*/, const std::vector<std::string>& /*options*/, sofa::simulation::Node::SPtr groot, const char* filename)
{

    glutInitDisplayMode ( GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE );

    //glutInitWindowPosition ( x0, y0 );
    //glutInitWindowSize ( nx, ny );
    glutCreateWindow ( ":: SOFA ::" );


#ifndef PS3
    std::cout << "Window created:"
            << " red="<<glutGet(GLUT_WINDOW_RED_SIZE)
            << " green="<<glutGet(GLUT_WINDOW_GREEN_SIZE)
            << " blue="<<glutGet(GLUT_WINDOW_BLUE_SIZE)
            << " alpha="<<glutGet(GLUT_WINDOW_ALPHA_SIZE)
            << " depth="<<glutGet(GLUT_WINDOW_DEPTH_SIZE)
            << " stencil="<<glutGet(GLUT_WINDOW_STENCIL_SIZE)
            << std::endl;
#endif

    glClearColor ( 0.0f, 0.0f, 0.0f, 0.0f );
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glutSwapBuffers ();
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glutSwapBuffers ();

    glutIdleFunc ( glut_idle );
    glutDisplayFunc ( glut_display );

#ifndef PS3
    glutReshapeFunc ( glut_reshape );
    glutKeyboardFunc ( glut_keyboard );
    glutSpecialFunc ( glut_special );
    glutMouseFunc ( glut_mouse );
    glutMotionFunc ( glut_motion );
    glutPassiveMotionFunc ( glut_motion );

    SimpleGUI* gui = new SimpleGUI();
    gui->setScene(groot, filename);

    gui->initializeGL();
#else
    // no glutReshape on PS3 the resoluion is fixed
    GLuint screen_width;
    GLuint screen_height;
    psglGetDeviceDimensions(psglGetCurrentDevice(), &screen_width, &screen_height);

    SimpleGUI* gui = new SimpleGUI();
    gui->setScene(groot, filename);

    gui->initializeGL();
    gui->resizeGL(screen_width, screen_height);
#endif

    return gui;
}



void SimpleGUI::glut_display()
{
    glClear ( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    if (instance)
    {
        instance->paintGL();
    }
    glutSwapBuffers ();
}

void SimpleGUI::glut_reshape(int w, int h)
{
    if (instance)
    {
        instance->resizeGL(w,h);
    }
}

void SimpleGUI::glut_keyboard(unsigned char k, int, int)
{
    if (instance)
    {
        instance->updateModifiers();
        instance->keyPressEvent(k);
    }
}

void SimpleGUI::glut_mouse(int button, int state, int x, int y)
{
    if (instance)
    {
        instance->updateModifiers();
        instance->mouseEvent( (state==GLUT_DOWN?MouseButtonPress : MouseButtonRelease), x, y, button );
    }
}

void SimpleGUI::glut_motion(int x, int y)
{
    if (instance)
    {
        //instance->updateModifiers();
        instance->mouseEvent( MouseMove, x, y, 0 );
    }
}

void SimpleGUI::glut_special(int k, int, int)
{
    if (instance)
    {
        instance->updateModifiers();
        instance->keyPressEvent(k);
    }
}

void SimpleGUI::glut_idle()
{
    if (instance)
    {
        if (instance->getScene() && instance->getScene()->getContext()->getAnimate())
            instance->step();
        else
            CTime::sleep(0.01);
        instance->animate();
    }
}



// ---------------------------------------------------------
// --- Constructor
// ---------------------------------------------------------
SimpleGUI::SimpleGUI()
{
    instance = this;

    groot = NULL;
    initTexturesDone = false;
    // setup OpenGL mode for the window

    _previousEyePos = Vector3(0.0, 0.0, 0.0);
    _zoom = 1.0;
    _zoomSpeed = 250.0;
    _panSpeed = 25.0;
    _navigationMode = 0;
    _spinning = false;
    _moving = false;
    _video = false;
    _animationOBJ = false;
    _axis = false;
    _background = 0;
    _numOBJmodels = 0;
    _materialMode = 0;
    _facetNormal = GL_FALSE;
    _renderingMode = GL_RENDER;
    _waitForRender = false;
    texLogo = NULL;

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

    ////////////////
    // Interactor //
    ////////////////
    _mouseInteractorMoving = false;
    _mouseInteractorSavedPosX = 0;
    _mouseInteractorSavedPosY = 0;

    //////////////////////
    m_isControlPressed = false;
    m_isShiftPressed = false;
    m_isAltPressed = false;
    m_dumpState = false;
    m_dumpStateStream = 0;
    m_exportGnuplot = false;

    //Register the different Operations possible
    RegisterOperation("Attach").add< AttachOperation >();
    RegisterOperation("Add recorded camera").add< AddRecordedCameraOperation >();
    RegisterOperation("Start navigation").add< StartNavigationOperation >();
    RegisterOperation("Fix").add< FixOperation >();
    RegisterOperation("Incise").add< InciseOperation >();
    RegisterOperation("Remove").add< TopologyOperation >();

    //Add to each button of the mouse an operation
    pick.changeOperation(LEFT,   "Attach");
    pick.changeOperation(MIDDLE, "Incise");
    pick.changeOperation(RIGHT,  "Remove");

    vparams = core::visual::VisualParams::defaultInstance();
    vparams->drawTool() = &drawTool;
}


// ---------------------------------------------------------
// --- Destructor
// ---------------------------------------------------------
SimpleGUI::~SimpleGUI()
{
    if (instance == this) instance = NULL;
}

// -----------------------------------------------------------------
// --- OpenGL initialization method - includes light definitions,
// --- color tracking, etc.
// -----------------------------------------------------------------
void SimpleGUI::initializeGL(void)
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
#if defined(SOFA_HAVE_GLEW) && !defined(PS3)
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
        glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specref);
        glMateriali(GL_FRONT_AND_BACK, GL_SHININESS, 128);

        glShadeModel(GL_SMOOTH);

        // Define background color
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        //Load texture for logo
        std::string filename = sofa::helper::system::DataRepository.getFile("textures/SOFA_logo.bmp");
        std::string extension = sofa::helper::system::SetDirectory::GetExtension(filename.c_str());
        std::transform(extension.begin(),extension.end(),extension.begin(),::tolower );
        bool imageSupport = helper::io::Image::FactoryImage::getInstance()->hasKey(extension);
        if(!imageSupport)
        {
            msg_error("SimpleGUI") << "Could not open sofa logo, " << extension  << " image format (no support found)" ;
            return;
        } else
        {

            helper::io::Image* img =  helper::io::Image::FactoryImage::getInstance()->createObject(extension, "");
            bool imgLoaded = img->load(filename);
            if (!imgLoaded)
            {
                msg_error("SimpleGUI") << "Could not open sofa logo, " << filename ;
                return;
            }
            texLogo = new helper::gl::Texture(img);
            texLogo->init();
        }

#ifndef PS3
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);
#endif
        // Turn on our light and enable color along with the light
        //glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        //glEnable(GL_COLOR_MATERIAL);

        // change status so we only do this stuff once
        initialized = true;

        _beginTime = CTime::getTime();

        printf("\n");
    }

    // switch to preset view
    resetView();
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void SimpleGUI::PrintString(void* font, char* string)
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
void SimpleGUI::Display3DText(float x, float y, float z, char* string)
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
void SimpleGUI::DrawAxis(double xpos, double ypos, double zpos,
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
void SimpleGUI::DrawBox(SReal* minBBox, SReal* maxBBox, double r)
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
void SimpleGUI::DrawXYPlane(double zo, double xmin, double xmax, double ymin,
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
void SimpleGUI::DrawYZPlane(double xo, double ymin, double ymax, double zmin,
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
void SimpleGUI::DrawXZPlane(double yo, double xmin, double xmax, double zmin,
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
void SimpleGUI::DrawLogo()
{
    int w = 0;
    int h = 0;

    if (texLogo && texLogo->getImage() && texLogo->getImage()->isLoaded())
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
void SimpleGUI::DisplayOBJs()
{

    Enable<GL_LIGHTING> light;
    Enable<GL_DEPTH_TEST> depth;

    glShadeModel(GL_SMOOTH);
    //glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColor4f(1,1,1,1);
    glDisable(GL_COLOR_MATERIAL);


    vparams->sceneBBox() = groot->f_bbox.getValue();

    if (!initTexturesDone)
    {
//         std::cout << "-----------------------------------> initTexturesDone\n";
        //---------------------------------------------------
        simulation::getSimulation()->initTextures(groot.get());
        //---------------------------------------------------
        initTexturesDone = true;
    }

    {

        getSimulation()->draw(vparams,groot.get());

        if (_axis)
        {
            DrawAxis(0.0, 0.0, 0.0, 10.0);
            if (vparams->sceneBBox().minBBox().x() < vparams->sceneBBox().maxBBox().x())
                DrawBox(vparams->sceneBBox().minBBoxPtr(),
                        vparams->sceneBBox().maxBBoxPtr());
        }
    }

    // glDisable(GL_COLOR_MATERIAL);
}

// -------------------------------------------------------
// ---
// -------------------------------------------------------
void SimpleGUI::DisplayMenu(void)
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
void SimpleGUI::DrawScene(void)
{
    if (!groot) return;
    if(!currentCamera)
    {
        std::cerr << "ERROR: no camera defined" << std::endl;
        return;
    }

    calcProjection();

    if (_background==0)
        DrawLogo();

    glLoadIdentity();

    GLdouble mat[16];

    currentCamera->getOpenGLModelViewMatrix(mat);
    glMultMatrixd(mat);

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


// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void SimpleGUI::resizeGL(int width, int height)
{

    _W = width;
    _H = height;

    if(currentCamera)
        currentCamera->setViewport(width, height);

//     std::cout << "GL window: " <<width<<"x"<<height <<std::endl;

    calcProjection();
}


// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void SimpleGUI::calcProjection()
{
    int width = _W;
    int height = _H;
    double xNear, yNear/*, xOrtho, yOrtho*/;
    double xFactor = 1.0, yFactor = 1.0;
    double offset;
    double xForeground, yForeground, zForeground, xBackground, yBackground,
           zBackground;
    Vector3 center;

    /// Camera part
    if (!currentCamera)
        return;

    if (groot && (!groot->f_bbox.getValue().isValid() || _axis))
    {
        vparams->sceneBBox() = groot->f_bbox.getValue();
        currentCamera->setBoundingBox(vparams->sceneBBox().minBBox(), vparams->sceneBBox().maxBBox());
    }
    currentCamera->computeZ();

    vparams->zNear() = currentCamera->getZNear();
    vparams->zFar() = currentCamera->getZFar();
    ///

    xNear = 0.35 * vparams->zNear();
    yNear = 0.35 * vparams->zNear();
    offset = 0.001 * vparams->zNear(); // for foreground and background planes

    /*xOrtho = fabs(vparams->sceneTransform().translation[2]) * xNear
            / vparams->zNear();
    yOrtho = fabs(vparams->sceneTransform().translation[2]) * yNear
            / vparams->zNear();*/

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
    vparams->viewport() = sofa::helper::make_array(0,0,width,height);

    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    xFactor *= 0.01;
    yFactor *= 0.01;

    //std::cout << xNear << " " << yNear << std::endl;

    zForeground = -vparams->zNear() - offset;
    zBackground = -vparams->zFar() + offset;

    if (currentCamera->getCameraType() == core::visual::VisualParams::PERSPECTIVE_TYPE)
        gluPerspective(currentCamera->getFieldOfView(), (double) width / (double) height, vparams->zNear(), vparams->zFar());
    else
    {
        float ratio = (float)( vparams->zFar() / (vparams->zNear() * 20) );
        Vector3 tcenter = vparams->sceneTransform() * center;
        if (tcenter[2] < 0.0)
        {
            ratio = (float)( -300 * (tcenter.norm2()) / tcenter[2] );
        }
        glOrtho((-xNear * xFactor) * ratio, (xNear * xFactor) * ratio, (-yNear
                * yFactor) * ratio, (yNear * yFactor) * ratio,
                vparams->zNear(), vparams->zFar());
    }

    xForeground = -zForeground * xNear / vparams->zNear();
    yForeground = -zForeground * yNear / vparams->zNear();
    xBackground = -zBackground * xNear / vparams->zNear();
    yBackground = -zBackground * yNear / vparams->zNear();

    xForeground *= xFactor;
    yForeground *= yFactor;
    xBackground *= xFactor;
    yBackground *= yFactor;

    glGetDoublev(GL_PROJECTION_MATRIX,lastProjectionMatrix);

    glMatrixMode(GL_MODELVIEW);
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void SimpleGUI::paintGL()
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
        glClearColor(0.0f,0.0f,0.0f,0.0f);
    //glClearColor(0.0589f, 0.0589f, 0.0589f, 1.0f);
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

    if (_waitForRender)
        _waitForRender = false;
}

void SimpleGUI::eventNewStep()
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
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void SimpleGUI::animate(void)
{
    if (_spinning)
    {
        //_newQuat = _currentQuat + _newQuat;
    }

    // update the entire scene
    redraw();
}


// ----------------------------------------
// --- Handle events (mouse, keyboard, ...)
// ----------------------------------------

bool SimpleGUI::isControlPressed() const
{
    return m_isControlPressed;
    //return glutGetModifiers()&GLUT_ACTIVE_CTRL;
}

bool SimpleGUI::isShiftPressed() const
{
    return m_isShiftPressed;
    //return glutGetModifiers()&GLUT_ACTIVE_SHIFT;
}

bool SimpleGUI::isAltPressed() const
{
    return m_isAltPressed;
    //return glutGetModifiers()&GLUT_ACTIVE_ALT;
}

void SimpleGUI::updateModifiers()
{
#ifndef PS3
    m_isControlPressed =  (glutGetModifiers()&GLUT_ACTIVE_CTRL )!=0;
    m_isShiftPressed   =  (glutGetModifiers()&GLUT_ACTIVE_SHIFT)!=0;
    m_isAltPressed     =  (glutGetModifiers()&GLUT_ACTIVE_ALT  )!=0;
#endif
}

void SimpleGUI::keyPressEvent ( int k )
{
#ifndef PS3
    if( isControlPressed() ) // pass event to the scene data structure
    {
        //cerr<<"SimpleGUI::keyPressEvent, key = "<<k<<" with Control pressed "<<endl;
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

        case GLUT_KEY_F5:
        {
            if (!sceneFileName.empty())
            {
                std::cout << "Reloading "<<sceneFileName<<std::endl;
                std::string filename = sceneFileName;
                Vec3d pos;
                Quat  ori;
                getView(pos, ori);

                simulation::Node::SPtr newroot = getSimulation()->load(filename.c_str());
                getSimulation()->init(newroot.get());
                if (newroot == NULL)
                {
                    std::cerr << "Failed to load "<<filename<<std::endl;
                    break;
                }
                setScene(newroot, filename.c_str());
                setView(pos, ori);
            }

            break;
        }
        }
#endif
}


void SimpleGUI::keyReleaseEvent ( int k )
{
    //cerr<<"SimpleGUI::keyReleaseEvent, key = "<<k<<endl;
    if( isControlPressed() ) // pass event to the scene data structure
    {
        sofa::core::objectmodel::KeyreleasedEvent keyEvent(k);
        groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
    }
}

// ---------------------- Here are the mouse controls for the scene  ----------------------
void SimpleGUI::mouseEvent ( int type, int eventX, int eventY, int button )
{
#ifndef PS3
    const sofa::core::visual::VisualParams::Viewport& viewport = vparams->viewport();

    MousePosition mousepos;
    mousepos.screenWidth  = viewport[2];
    mousepos.screenHeight = viewport[3];
    mousepos.x      = eventX;
    mousepos.y      = eventY;

    if( isShiftPressed() )
    {
        pick.activateRay(viewport[2],viewport[3], groot.get());
    }
    else
    {
        pick.deactivateRay();
    }

    if (isShiftPressed())
    {
        _moving = false;

        Vec3d p0, px, py, pz;
        gluUnProject(eventX, viewport[3]-1-(eventY), 0, lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(p0[0]), &(p0[1]), &(p0[2]));
        gluUnProject(eventX+1, viewport[3]-1-(eventY), 0, lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(px[0]), &(px[1]), &(px[2]));
        gluUnProject(eventX, viewport[3]-1-(eventY+1), 0, lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(py[0]), &(py[1]), &(py[2]));
        gluUnProject(eventX, viewport[3]-1-(eventY), 1, lastModelviewMatrix, lastProjectionMatrix, viewport.data(), &(pz[0]), &(pz[1]), &(pz[2]));
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
                _lightPosition[0] -= dx*0.1f;
                _lightPosition[1] += dy*0.1f;
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
                //g_DepthBias[1] += dy*0.01;
                //std::cout << "Depth bias = "<< g_DepthBias[0] << " " << g_DepthBias[1] << std::endl;
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
                //g_DepthOffset[0] += dx*0.01;
                //g_DepthOffset[1] += dy*0.01;
                //std::cout << "Depth offset = "<< g_DepthOffset[0] << " " << g_DepthOffset[1] << std::endl;
                redraw();
                _mouseInteractorSavedPosX = eventX;
                _mouseInteractorSavedPosY = eventY;
            }
        }
    }
    else if (isControlPressed())
    {

    }
    else
    {
        switch (type)
        {
        case MouseButtonPress:
        {
            //<CAMERA API>
            sofa::core::objectmodel::MouseEvent* mEvent = NULL;
            if (button == GLUT_LEFT_BUTTON)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed, eventX, eventY);
            else if (button == GLUT_RIGHT_BUTTON)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed, eventX, eventY);
            else if (button == GLUT_MIDDLE_BUTTON)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed, eventX, eventY);
            else{
                // A fallback event to rules them all...
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::AnyExtraButtonPressed, eventX, eventY);
            }
            currentCamera->manageEvent(mEvent);
            _moving = true;
            _spinning = false;
            _mouseX = eventX;
            _mouseY = eventY;
            break;
        }
        case MouseMove:
        {
            //<CAMERA API>
            sofa::core::objectmodel::MouseEvent me(sofa::core::objectmodel::MouseEvent::Move, eventX, eventY);
            currentCamera->manageEvent(&me);
            break;
        }

        case MouseButtonRelease:
        {
            //<CAMERA API>
            sofa::core::objectmodel::MouseEvent* mEvent = NULL;
            if (button == GLUT_LEFT_BUTTON)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased, eventX, eventY);
            else if (button == GLUT_RIGHT_BUTTON)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased, eventX, eventY);
            else if (button == GLUT_MIDDLE_BUTTON)
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased, eventX, eventY);
            else{
                // A fallback event to rules them all...
                mEvent = new sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::AnyExtraButtonReleased, eventX, eventY);
            }
            currentCamera->manageEvent(mEvent);
            _moving = false;
            _spinning = false;
            _mouseX = eventX;
            _mouseY = eventY;
            break;
        }

        default:
            break;
        }

        redraw();
    }
#endif
}

void SimpleGUI::step()
{
    {
        if (_waitForRender) return;
        //groot->setLogTime(true);
#ifdef SOFA_SMP
        mg->step();
#else
        getSimulation()->animate(groot.get());
#endif
        getSimulation()->updateVisual(groot.get());

        if( m_dumpState )
            getSimulation()->dumpState( groot.get(), *m_dumpStateStream );
        if( m_exportGnuplot )


            _waitForRender = true;
        eventNewStep();

        redraw();
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

void SimpleGUI::playpause()
{
    if (groot)
    {
        groot->getContext()->setAnimate(!groot->getContext()->getAnimate());
    }
}

void SimpleGUI::dumpState(bool value)
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

void SimpleGUI::resetScene()
{
    if (groot)
    {
        getSimulation()->reset(groot.get());
        redraw();
    }
}

void SimpleGUI::resetView()
{
    bool fileRead = false;

    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName + ".view";
        fileRead = currentCamera->importParametersFromFile(viewFileName);
    }

    //if there is no .view file , look at the center of the scene bounding box
    // and with a Up vector in the same axis as the gravity
    if (!fileRead)
    {
        newView();
    }
    redraw();
}

void SimpleGUI::setCameraMode(core::visual::VisualParams::CameraType mode)
{
    currentCamera->setCameraType(mode);
}

void SimpleGUI::getView(Vec3d& pos, Quat& ori) const
{
    if (!currentCamera)
        return;

    const Vec3d& camPosition = currentCamera->getPosition();
    const Quat& camOrientation = currentCamera->getOrientation();

    pos[0] = camPosition[0];
    pos[1] = camPosition[1];
    pos[2] = camPosition[2];

    ori[0] = camOrientation[0];
    ori[1] = camOrientation[1];
    ori[2] = camOrientation[2];
    ori[3] = camOrientation[3];
}

void SimpleGUI::setView(const Vec3d& pos, const Quat &ori)
{
    Vec3d position;
    Quat orientation;
    for (unsigned int i=0 ; i<3 ; i++)
    {
        position[i] = pos[i];
        orientation[i] = ori[i];
    }
    orientation[3] = ori[3];

    if (currentCamera)
        currentCamera->setView(position, orientation);

    redraw();
}

void SimpleGUI::moveView(const Vec3d& pos, const Quat &ori)
{
    if (!currentCamera)
        return;

    currentCamera->moveCamera(pos, ori);
    redraw();
}

void SimpleGUI::newView()
{
    if (!currentCamera || !groot)
        return;

    currentCamera->setDefaultView(groot->getGravity());
}

void SimpleGUI::saveView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName + ".view";

        if(currentCamera->exportParametersInFile(viewFileName))
            std::cout << "View parameters saved in " << viewFileName << std::endl;
        else
            std::cout << "Error while saving view parameters in " << viewFileName << std::endl;
    }
}

void SimpleGUI::screenshot(int compression_level)
{
    capture.saveScreen(compression_level);
}

void SimpleGUI::exportOBJ(bool exportMTL)
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
    getSimulation()->exportOBJ(groot.get(), filename.c_str(),exportMTL);
}

void SimpleGUI::setScene(sofa::simulation::Node::SPtr scene, const char* filename, bool)
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

    //Camera initialization
    if (groot)
    {
        groot->get(currentCamera);
        if (!currentCamera)
        {
            currentCamera = sofa::core::objectmodel::New<component::visualmodel::InteractiveCamera>();
            currentCamera->setName(core::objectmodel::Base::shortName(currentCamera.get()));
            groot->addObject(currentCamera);
            currentCamera->p_position.forceSet();
            currentCamera->p_orientation.forceSet();
            currentCamera->bwdInit();
            resetView();
        }

        vparams->sceneBBox() = groot->f_bbox.getValue();
        currentCamera->setBoundingBox(vparams->sceneBBox().minBBox(), vparams->sceneBBox().maxBBox());

        // init pickHandler
        pick.init(groot.get());
    }
    redraw();
}

void SimpleGUI::setExportGnuplot( bool exp )
{
    m_exportGnuplot = exp;
    if( m_exportGnuplot )
    {
        exportGnuplot(groot.get());
    }
}

} // namespace glut

} // namespace gui

} // namespace sofa
