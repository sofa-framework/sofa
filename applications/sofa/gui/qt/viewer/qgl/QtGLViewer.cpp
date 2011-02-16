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
#include "viewer/qgl/QtGLViewer.h"
#include <sofa/helper/system/config.h>
#include <sofa/helper/system/FileRepository.h>
#include <sofa/helper/system/thread/CTime.h>
#include <sofa/simulation/common/Simulation.h>
#include <sofa/simulation/common/MechanicalVisitor.h>
#include <sofa/simulation/common/UpdateMappingVisitor.h>
#include <sofa/core/objectmodel/KeypressedEvent.h>
#include <sofa/core/objectmodel/KeyreleasedEvent.h>
#include <sofa/core/ObjectFactory.h>
#include <sofa/simulation/common/ColourPickingVisitor.h>
//#include <sofa/helper/system/SetDirectory.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <sofa/helper/system/gl.h>
#include <sofa/helper/system/glu.h>
#include <sofa/helper/system/glut.h>
#include <sofa/gui/SofaGUI.h>
#include <qevent.h>
#include "GenGraphForm.h"

#include <sofa/helper/gl/DrawManagerGL.h>

#include <sofa/helper/gl/glfont.h>
#include <sofa/helper/gl/RAII.h>
#ifdef SOFA_HAVE_GLEW
#include <sofa/helper/gl/GLSLShader.h>
#endif
#include <sofa/helper/io/ImageBMP.h>

#include <sofa/defaulttype/RigidTypes.h>


// define this if you want video and OBJ capture to be only done once per N iteration
//#define CAPTURE_PERIOD 5


namespace sofa
{

namespace gui
{

namespace qt
{

namespace viewer
{

namespace qgl
{

using std::cout;
using std::endl;
using namespace sofa::defaulttype;
using namespace sofa::helper::gl;
using sofa::simulation::getSimulation;
using namespace sofa::simulation;

helper::SofaViewerCreator<QtGLViewer> QtGLViewer_class("qglviewer",false);
SOFA_DECL_CLASS ( QGLViewerGUI )


static bool LeftPressedForMove = false;
static bool RightPressedForMove = false;


// ---------------------------------------------------------
// --- Constructor
// ---------------------------------------------------------
QtGLViewer::QtGLViewer(QWidget* parent, const char* name)
    : QGLViewer(parent, name)
{
    sofa::simulation::getSimulation()->setDrawUtility(new sofa::helper::gl::DrawManagerGL);

    groot = NULL;
    initTexturesDone = false;

#ifdef TRACKING_MOUSE
    m_grabActived = false;
#endif

    backgroundColour[0]=1.0f;
    backgroundColour[1]=1.0f;
    backgroundColour[2]=1.0f;

    // setup OpenGL mode for the window
    //Fl_Gl_Window::mode(FL_RGB | FL_DOUBLE | FL_DEPTH | FL_ALPHA);
    timerAnimate = new QTimer(this);
    connect( timerAnimate, SIGNAL(timeout()), this, SLOT(animate()) );

    //	_previousEyePos = Vector3(0.0, 0.0, 0.0);
    // 	_zoom = 1.0;
    // 	_zoomSpeed = 250.0;
    // 	_panSpeed = 25.0;
    _video = false;
    _axis = false;
    _background = 0;
    _numOBJmodels = 0;
    _materialMode = 0;
    _facetNormal = GL_FALSE;
    _renderingMode = GL_RENDER;

    sceneBBoxIsValid = false;
    _waitForRender=false;

    /*_surfaceModel = NULL;
    _springMassView = NULL;
    _mapView = NULL;
    sphViewer = NULL;
    */

    //////////////////////

    _mouseInteractorMoving = false;
    _mouseInteractorSavedPosX = 0;
    _mouseInteractorSavedPosY = 0;
    m_isControlPressed = false;

    setManipulatedFrame( new qglviewer::ManipulatedFrame() );
    //near and far plane are better placed
    camera()->setZNearCoefficient(0.001);
    camera()->setZClippingCoefficient(5);

    visualParameters.zNear = camera()->zNear();
    visualParameters.zFar = camera()->zFar();

    connect( &captureTimer, SIGNAL(timeout()), this, SLOT(captureEvent()) );
}


// ---------------------------------------------------------
// --- Destructor
// ---------------------------------------------------------
QtGLViewer::~QtGLViewer()
{
}

// -----------------------------------------------------------------
// --- OpenGL initialization method - includes light definitions,
// --- color tracking, etc.
// -----------------------------------------------------------------
void QtGLViewer::init(void)
{
    restoreStateFromFile();


    static	 GLfloat	specref[4];
    static	 GLfloat	ambientLight[4];
    static	 GLfloat	diffuseLight[4];
    static	 GLfloat	specular[4];
    static	 GLfloat	lmodel_ambient[]	= {0.0f, 0.0f, 0.0f, 0.0f};
    static	 GLfloat	lmodel_twoside[]	= {GL_FALSE};
    static	 GLfloat	lmodel_local[]		= {GL_FALSE};
    bool		initialized			= false;

    if (!initialized)
    {
        //std::cout << "progname="<<sofa::gui::qt::progname<<std::endl;
        //sofa::helper::system::SetDirectory cwd(sofa::helper::system::SetDirectory::GetProcessFullPath(sofa::gui::qt::progname));

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
#endif
#if 0
        if (!GLEW_ARB_multitexture)
            std::cerr << "Error: GL_ARB_multitexture not supported\n";

        glActiveTextureARB        = (PFNGLACTIVETEXTUREARBPROC)        glewGetProcAddress("glActiveTextureARB");
        glMultiTexCoord2fARB    = (PFNGLMULTITEXCOORD2FARBPROC)        glewGetProcAddress("glMultiTexCoord2fARB");

        // Make sure our multi-texturing extensions were loaded correctly
        if(!glActiveTextureARB || !glMultiTexCoord2fARB)
        {
            // Print an error message and quit.
            //    MessageBox(g_hWnd, "Your current setup does not support multitexturing", "Error", MB_OK);
            //PostQuitMessage(0);
        }
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
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

        //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
        //Load texture for logo
        setBackgroundImage();


        glEnableClientState(GL_VERTEX_ARRAY);
        //glEnableClientState(GL_NORMAL_ARRAY);

        // Turn on our light and enable color along with the light
        //glEnable(GL_LIGHTING);
        glEnable(GL_LIGHT0);
        //glEnable(GL_COLOR_MATERIAL);

        //init Quadrics
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

        // change status so we only do this stuff once
        initialized = true;

        _beginTime = CTime::getTime();

        printf("\n");


        // GL_LIGHT1 follows the camera
        // 	glMatrixMode(GL_MODELVIEW);
        // 	glPushMatrix();
        // 	glLoadIdentity();
        // 	glLightfv(GL_LIGHT0, GL_POSITION, _lightPosition);
        // 	glPopMatrix();



        // 	camera()->setType( qglviewer::Camera::ORTHOGRAPHIC );
        // 	camera()->setType( qglviewer::Camera::PERSPECTIVE  );

    }

    // switch to preset view

    resetView();

    // Redefine keyboard events
    // The default SAVE_SCREENSHOT shortcut is Ctrl+S and this shortcut is used to
    // save x3d file in the MainController. So we need to change it:
    setShortcut(QGLViewer::SAVE_SCREENSHOT, Qt::Key_S);
    setShortcut(QGLViewer::HELP, Qt::Key_H);

}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtGLViewer::PrintString(void* font, char* string)
{
    int	len, i;

    len = (int) strlen(string);
    for (i = 0; i < len; i++)
    {
        glutBitmapCharacter(font, string[i]);
    }
}

// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtGLViewer::Display3DText(float x, float y, float z, char* string)
{
    char*	c;

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
void QtGLViewer::DrawAxis(double xpos, double ypos, double zpos,
        double arrowSize)
{
    glPushMatrix();
    glTranslatef(xpos, ypos,zpos);
    QGLViewer::drawAxis(arrowSize);
    glPopMatrix();
}

// ---------------------------------------------------
// ---
// ---
// ---------------------------------------------------
void QtGLViewer::DrawBox(Real* minBBox, Real* maxBBox, Real r)
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
void QtGLViewer::DrawXYPlane(double zo, double xmin, double xmax, double ymin,
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
void QtGLViewer::DrawYZPlane(double xo, double ymin, double ymax, double zmin,
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
void QtGLViewer::DrawXZPlane(double yo, double xmin, double xmax, double zmin,
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

void QtGLViewer::drawColourPicking(core::CollisionModel::ColourCode code)
{

    // Define background color
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // GL_PROJECTION matrix
    camera()->loadProjectionMatrix();
    // GL_MODELVIEW matrix
    camera()->loadModelViewMatrix();



    ColourPickingVisitor cpv(sofa::core::ExecParams::defaultInstance(), code);
    cpv.execute(sofa::simulation::getSimulation()->getContext() );

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void QtGLViewer::DrawLogo()
{
    glPushMatrix();

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
    glPopMatrix();
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void QtGLViewer::DisplayOBJs()
{
    if (!groot) return;

    if (!sceneBBoxIsValid) viewAll();

    Enable<GL_LIGHTING> light;
    Enable<GL_DEPTH_TEST> depth;

    glShadeModel(GL_SMOOTH);
    //glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    glColor4f(1,1,1,1);
    glDisable(GL_COLOR_MATERIAL);

    if (!initTexturesDone)
    {
        //		std::cout << "-----------------------------------> initTexturesDone\n";
        //---------------------------------------------------
        simulation::getSimulation()->initTextures(groot);
        //---------------------------------------------------
        initTexturesDone = true;
    }

    {
        //Draw Debug information of the components
        simulation::getSimulation()->draw(groot, &visualParameters);
        //Draw Visual Models
        simulation::getSimulation()->draw(simulation::getSimulation()->getVisualRoot(), &visualParameters);
        if (_axis)
        {
            this->setSceneBoundingBox(qglviewer::Vec(visualParameters.minBBox[0], visualParameters.minBBox[1], visualParameters.minBBox[2]),
                    qglviewer::Vec(visualParameters.maxBBox[0], visualParameters.maxBBox[1], visualParameters.maxBBox[2]));

            //DrawAxis(0.0, 0.0, 0.0, 10.0);
            DrawAxis(0.0, 0.0, 0.0, this->sceneRadius());

            if (visualParameters.minBBox[0] < visualParameters.maxBBox[0])
                DrawBox(visualParameters.minBBox.ptr(), visualParameters.maxBBox.ptr());
        }
    }

    // glDisable(GL_COLOR_MATERIAL);
}

// -------------------------------------------------------
// ---
// -------------------------------------------------------
void QtGLViewer::DisplayMenu(void)
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
void QtGLViewer::DrawScene(void)
{

    camera()->getProjectionMatrix( lastProjectionMatrix );

    camera()->getViewport( visualParameters.viewport );
    visualParameters.viewport[1]=0;
    visualParameters.viewport[3]=-visualParameters.viewport[3];




    if (_background==0)
        DrawLogo();


    camera()->getModelViewMatrix( lastModelviewMatrix );
    //camera()->frame()->getMatrix( lastModelviewMatrix );

    //for(int i=0 ; i<16 ;i++)
    //	std::cout << lastModelviewMatrix[i] << " ";
    //
    //std::cout << std::endl;
    //std::cout << "P " << camera()->position().x << " " << camera()->position().y << " " << camera()->position().z << " " << std::endl;
    //std::cout << "T " << camera()->frame()->translation().x << " " << camera()->frame()->translation().y << " " << camera()->frame()->translation().z << " " << std::endl;
    //std::cout << "Q " << camera()->orientation() << std::endl;
    //std::cout << "R " << camera()->frame()->rotation() << " " << std::endl;

    if (_renderingMode == GL_RENDER)
    {
        // 		// Initialize lighting
        glPushMatrix();
        glLoadIdentity();
        glLightfv(GL_LIGHT0, GL_POSITION, _lightPosition);
        glPopMatrix();
        Enable<GL_LIGHT0> light0;
        //
        glColor3f(0.5f, 0.5f, 0.6f);
        // 			DrawXZPlane(-4.0, -20.0, 20.0, -20.0, 20.0, 1.0);
        // 			DrawAxis(0.0, 0.0, 0.0, 10.0);

        DisplayOBJs();

        DisplayMenu();		// always needs to be the last object being drawn
    }

}

void QtGLViewer::viewAll()
{
    if (!groot) return;

    getSimulation()->computeBBox(groot, visualParameters.minBBox.ptr(), visualParameters.maxBBox.ptr());
    getSimulation()->computeBBox(getSimulation()->getVisualRoot(), visualParameters.minBBox.ptr(), visualParameters.maxBBox.ptr(),false);

    sceneBBoxIsValid =
        visualParameters.minBBox[0]    <=  visualParameters.maxBBox[0]
        && visualParameters.minBBox[1] <=  visualParameters.maxBBox[1]
        && visualParameters.minBBox[2] <=  visualParameters.maxBBox[2];

    if (visualParameters.minBBox[0] == visualParameters.maxBBox[0])
    {
        visualParameters.minBBox[0]=-1;
        visualParameters.minBBox[0]= 1;
    }
    if (visualParameters.minBBox[1] == visualParameters.maxBBox[1])
    {
        visualParameters.minBBox[1]=-1;
        visualParameters.minBBox[1]= 1;
    }
    if (visualParameters.minBBox[2] == visualParameters.maxBBox[2])
    {
        visualParameters.minBBox[2]=-1;
        visualParameters.minBBox[2]= 1;
    }

    if (sceneBBoxIsValid) QGLViewer::setSceneBoundingBox(   qglviewer::Vec(visualParameters.minBBox.ptr()),qglviewer::Vec(visualParameters.maxBBox.ptr()) );

    qglviewer::Vec pos;
    pos[0] = 0.0;
    pos[1] = 0.0;
    pos[2] = 75.0;
    camera()->setPosition(pos);
    camera()->showEntireScene();
}



// ---------------------------------------------------------
// --- Reshape of the window, reset the projection
// ---------------------------------------------------------
void QtGLViewer::resizeGL(int width, int height)
{
    _W = width;
    _H = height;


    QGLViewer::resizeGL( width,  height);
    // 	    camera()->setScreenWidthAndHeight(_W,_H);

    this->resize(width, height);
    emit( resizeW( _W ) );
    emit( resizeH( _H ) );
}


// ---------------------------------------------------------
// ---
// ---------------------------------------------------------
void QtGLViewer::draw()
{
    //	ctime_t beginDisplay;
    //ctime_t endOfDisplay;

    //	beginDisplay = MesureTemps();

    // valid() is turned off when FLTK creates a new context for this window
    // or when the window resizes, and is turned on after draw() is called.
    // Use this to avoid unneccessarily initializing the OpenGL context.
    //static double lastOrthoTransZ = 0.0;
    /*
    if (!valid())
    {
    InitGFX();		// this has to be called here since we don't know when the context is created
    _W = w();
    _H = h();
    reshape(_W, _H);
    }
    */
    // clear buffers (color and depth)
    if (_background==0)
        glClearColor(0.0f,0.0f,0.0f,1.0f);
    else if (_background==1)
        glClearColor(0.0f,0.0f,0.0f,0.0f);
    else if (_background==2)
        glClearColor(backgroundColour[0],backgroundColour[1],backgroundColour[2], 1.0f);
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
        }

    SofaViewer::captureEvent();

    if (_waitForRender)
        _waitForRender = false;

    emit( redrawn() );
}

void QtGLViewer::setCameraMode(component::visualmodel::BaseCamera::CameraType mode)
{
    SofaViewer::setCameraMode(mode);

    switch (mode)
    {
    case component::visualmodel::BaseCamera::ORTHOGRAPHIC_TYPE:
        camera()->setType( qglviewer::Camera::ORTHOGRAPHIC );
        break;
    case component::visualmodel::BaseCamera::PERSPECTIVE_TYPE:
        camera()->setType( qglviewer::Camera::PERSPECTIVE  );
        break;
    }
}


// ----------------------------------------
// --- Handle events (mouse, keyboard, ...)
// ----------------------------------------


void QtGLViewer::keyPressEvent ( QKeyEvent * e )
{

    //Tracking Mode
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        if (e->key() == Qt::Key_Escape)
        {
            m_grabActived = false;
            this->setCursor(QCursor(Qt::ArrowCursor));
        }
        else
        {
            if (groot)
            {
                sofa::core::objectmodel::KeypressedEvent keyEvent(e->key());
                groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
            }
        }
        return;
    }
#endif

    // 	cerr<<"QtGLViewer::keyPressEvent, get "<<e->key()<<endl;
    if( isControlPressed() ) // pass event to the scene data structure
    {
#ifdef TRACKING_MOUSE
        if (e->key() == Qt::Key_T)
        {
            m_grabActived = true;
            this->setCursor(QCursor(Qt::BlankCursor));
            QPoint p = mapToGlobal(this->pos()) + QPoint((this->width()+2)/2,(this->height()+2)/2);
            QCursor::setPos(p);
        }
        else
#endif
            //cerr<<"QtGLViewer::keyPressEvent, key = "<<e->key()<<" with Control pressed "<<endl;
            if (groot)
            {
                sofa::core::objectmodel::KeypressedEvent keyEvent(e->key());
                groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
            }
    }
    else  // control the GUI
    {
//                            cerr<<"QtGLViewer::keyPressEvent, key = "<<e->key()<<" without Control pressed "<<endl;
        switch(e->key())
        {
        case Qt::Key_A: //axis
        case Qt::Key_S: //sofa screenshot
        case Qt::Key_H: //shortcuts for screenshot and help page specified for qglviewer
        {
            QGLViewer::keyPressEvent(e);
            break;
        }
        case Qt::Key_C:
        {
            viewAll();
            break;
        }
        default:
        {
            SofaViewer::keyPressEvent(e);
            QGLViewer::keyPressEvent(e);
            e->ignore();
        }
        }
    }
    update();
}





void QtGLViewer::keyReleaseEvent ( QKeyEvent * e )
{

#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        sofa::core::objectmodel::KeyreleasedEvent keyEvent(e->key());
        if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &keyEvent);
        return;
    }
#endif
    QGLViewer::keyReleaseEvent(e);
    SofaViewer::keyReleaseEvent(e);
}






void QtGLViewer::mousePressEvent ( QMouseEvent * e )
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        if (e->type() == QEvent::MouseButtonPress)
        {
            if (e->button() == Qt::LeftButton)
            {
                LeftPressedForMove = true;
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftPressed);
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::RightButton)
            {
                RightPressedForMove = true;
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightPressed);
                if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::MidButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddlePressed);
                if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            return;
        }
    }
#endif
    if( ! mouseEvent(e) )
        QGLViewer::mousePressEvent(e);
}




void QtGLViewer::mouseReleaseEvent ( QMouseEvent * e )
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        if (e->type() == QEvent::MouseButtonRelease)
        {
            if (e->button() == Qt::LeftButton)
            {
                LeftPressedForMove = false;
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::LeftReleased);
                if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::RightButton)
            {
                RightPressedForMove = false;
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::RightReleased);
                if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            else if (e->button() == Qt::MidButton)
            {
                sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::MiddleReleased);
                if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
            }
            return;
        }
    }
#endif
    if( ! mouseEvent(e) )
        QGLViewer::mouseReleaseEvent(e);
}





void QtGLViewer::mouseMoveEvent ( QMouseEvent * e )
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        QPoint p = mapToGlobal(this->pos()) + QPoint((this->width()+2)/2,(this->height()+2)/2);
        QPoint c = QCursor::pos();

        sofa::core::objectmodel::MouseEvent mouseEvent1(sofa::core::objectmodel::MouseEvent::Move,c.x() - p.x(),c.y() - p.y());
        sofa::core::objectmodel::MouseEvent mouseEvent2(sofa::core::objectmodel::MouseEvent::Move,p.x() - p.x(),c.y() - p.y());
        sofa::core::objectmodel::MouseEvent mouseEvent3(sofa::core::objectmodel::MouseEvent::Move,c.x() - p.x(),p.y() - p.y());


        QCursor::setPos(p);
        if((LeftPressedForMove == false && RightPressedForMove == false) || (LeftPressedForMove == true && RightPressedForMove == true))
            if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent1);
        if(LeftPressedForMove == true)
            if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent2);
        if(RightPressedForMove == true)
            if (groot)groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent3);
        return;
    }
#endif

    if( ! mouseEvent(e) )
        QGLViewer::mouseMoveEvent(e);
}





bool QtGLViewer::mouseEvent(QMouseEvent * e)
{
    if(e->state()&Qt::ShiftButton)
    {
        SofaViewer::mouseEvent(e);
        return true;
    }
    else if (e->state()&Qt::ControlButton)
    {
        moveLaparoscopic(e);
        return true;
    }

    return false;
}

void QtGLViewer::wheelEvent(QWheelEvent* e)
{
#ifdef TRACKING_MOUSE
    if(m_grabActived)
    {
        sofa::core::objectmodel::MouseEvent mouseEvent = sofa::core::objectmodel::MouseEvent(sofa::core::objectmodel::MouseEvent::Wheel,e->delta());
        if (groot) groot->propagateEvent(core::ExecParams::defaultInstance(), &mouseEvent);
        return;
    }
#endif
    if (e->state()&Qt::ControlButton)
        moveLaparoscopic(e);
    else
        QGLViewer::wheelEvent(e);
}

void QtGLViewer::moveRayPickInteractor(int eventX, int eventY)
{
    Vec3d p0, px, py, pz, px1, py1;
    gluUnProject(eventX,   visualParameters.viewport[3]-1-(eventY),   0,   lastModelviewMatrix, lastProjectionMatrix, visualParameters.viewport, &(p0[0]),  &(p0[1]),  &(p0[2]));
    gluUnProject(eventX+1, visualParameters.viewport[3]-1-(eventY),   0,   lastModelviewMatrix, lastProjectionMatrix, visualParameters.viewport, &(px[0]),  &(px[1]),  &(px[2]));
    gluUnProject(eventX,   visualParameters.viewport[3]-1-(eventY+1), 0,   lastModelviewMatrix, lastProjectionMatrix, visualParameters.viewport, &(py[0]),  &(py[1]),  &(py[2]));
    gluUnProject(eventX,   visualParameters.viewport[3]-1-(eventY),   0.1, lastModelviewMatrix, lastProjectionMatrix, visualParameters.viewport, &(pz[0]),  &(pz[1]),  &(pz[2]));
    gluUnProject(eventX+1, visualParameters.viewport[3]-1-(eventY),   0.1, lastModelviewMatrix, lastProjectionMatrix, visualParameters.viewport, &(px1[0]), &(px1[1]), &(px1[2]));
    gluUnProject(eventX,   visualParameters.viewport[3]-1-(eventY+1), 0,   lastModelviewMatrix, lastProjectionMatrix, visualParameters.viewport, &(py1[0]), &(py1[1]), &(py1[2]));
    px1 -= pz;
    py1 -= pz;
    px -= p0;
    py -= p0;
    pz -= p0;
    double r0 = sqrt(px.norm2() + py.norm2());
    double r1 = sqrt(px1.norm2() + py1.norm2());
    r1 = r0 + (r1-r0) / pz.norm();
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
}

// -------------------------------------------------------------------
// ---
// -------------------------------------------------------------------
void QtGLViewer::resetView()
{
    viewAll();

    if (!sceneFileName.empty())
    {
        //Test if we have a specific view point for the QGLViewer
        //That case, the camera will be well placed
        std::string viewFileName = sceneFileName+"."+sofa::gui::SofaGUI::GetGUIName()+".view";
        std::ifstream in(viewFileName.c_str());
        if (!in.fail())
        {
            qglviewer::Vec pos;
            in >> pos[0];
            in >> pos[1];
            in >> pos[2];

            qglviewer::Quaternion q;
            in >> q[0];
            in >> q[1];
            in >> q[2];
            in >> q[3];
            q.normalize();

            camera()->setOrientation(q);
            camera()->setPosition(pos);

            in.close();
            update();

            return;
        }
        else
        {
            //If we have the default QtViewer view file, we have to use, showEntireScene
            //as the FOV of the QtViewer is not constant, so the parameters are not good
            std::string viewFileName = sceneFileName+".view";
            std::ifstream in(viewFileName.c_str());
            if (!in.fail())
            {
                qglviewer::Vec pos;
                in >> pos[0];
                in >> pos[1];
                in >> pos[2];

                qglviewer::Quaternion q;
                in >> q[0];
                in >> q[1];
                in >> q[2];
                in >> q[3];
                q.normalize();

                camera()->setOrientation(q);
                camera()->setPosition(pos);
                camera()->showEntireScene();

                in.close();
                update();

                return;
            }
        }
    }
    update();
}

void QtGLViewer::saveView()
{
    if (!sceneFileName.empty())
    {
        std::string viewFileName = sceneFileName+"."+sofa::gui::SofaGUI::GetGUIName()+".view";
        std::ofstream out(viewFileName.c_str());
        if (!out.fail())
        {
            out << camera()->position()[0] << " " << camera()->position()[1] << " " << camera()->position()[2] << "\n";
            out << camera()->orientation()[0] << " " << camera()->orientation()[1] << " " << camera()->orientation()[2] << " " << camera()->orientation()[3] << "\n";
            out.close();
        }
        std::cout << "View parameters saved in "<<viewFileName<<std::endl;
    }
}

void QtGLViewer::setSizeW( int size )
{
    resizeGL( size, _H );
    updateGL();
}

void QtGLViewer::setSizeH( int size )
{
    resizeGL( _W, size );
    updateGL();

}


QString QtGLViewer::helpString()
{

    QString text(
        "<H1>QtGLViewer</H1><hr>\
                                <ul>\
                                <li><b>Mouse</b>: TO NAVIGATE<br></li>\
                                <li><b>Shift & Left Button</b>: TO PICK OBJECTS<br></li>\
                                <li><b>A</b>: TO DRAW AXIS<br></li>\
                                <li><b>B</b>: TO CHANGE THE BACKGROUND<br></li>\
                                <li><b>C</b>: TO CENTER THE VIEW<br></li>\
                                <li><b>H</b>: TO OPEN HELP of QGLViewer<br></li>\
                                <li><b>L</b>: TO DRAW SHADOWS<br></li>\
                                <li><b>O</b>: TO EXPORT TO .OBJ<br>\
                                The generated files scene-time.obj and scene-time.mtl are saved in the running project directory<br></li>\
                                <li><b>P</b>: TO SAVE A SEQUENCE OF OBJ<br>\
                                Each time the frame is updated an obj is exported<br></li>\
                                <li><b>R</b>: TO DRAW THE SCENE AXIS<br></li>\
                                <li><b>T</b>: TO CHANGE BETWEEN A PERSPECTIVE OR AN ORTHOGRAPHIC CAMERA<br></li>\
                                The captured images are saved in the running project directory under the name format capturexxxx.bmp<br></li>\
                                <li><b>S</b>: TO SAVE A SCREENSHOT<br>\
                                <li><b>V</b>: TO SAVE A VIDEO<br>\
                                Each time the frame is updated a screenshot is saved<br></li>\
                                <li><b>Esc</b>: TO QUIT ::sofa:: <br></li></ul>");

    return text;
}





} // namespace qgl

} // namespace viewer

} //namespace qt

} // namespace gui

} // namespace sofa
